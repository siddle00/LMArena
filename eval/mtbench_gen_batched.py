#!/usr/bin/env python3
# MT-Bench: DPO generations (batched, SFT-parity config)

import os, json, random
import pandas as pd
from tqdm.auto import tqdm
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ---------------------
# Config (env overrides OK)
# ---------------------
BASE_MODEL        = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
DPO_ADAPTER_PATH  = os.environ.get("DPO_ADAPTER_PATH", "./dpo_arena55k_0830_dulcet_glade_12")
OUT_DIR           = os.environ.get("OUT_DIR", "mtbench_runs")
OUT_FILE          = os.environ.get("OUT_FILE", "dpo_generations_batched.jsonl")


MAX_NEW_TOKENS    = int(os.environ.get("MAX_NEW_TOKENS", 1000))
TEMPERATURE       = float(os.environ.get("TEMPERATURE", 0.2))
TOP_P             = float(os.environ.get("TOP_P", 0.95))
BATCH_SIZE        = int(os.environ.get("BATCH_SIZE", 16))


SEED              = int(os.environ.get("SEED", 42))
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16          = torch.cuda.is_available() and torch.cuda.is_bf16_supported()


# ---------------------
# Repro & perf knobs
# ---------------------
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ---------------------
# Data: MT-Bench prompts
# ---------------------
mtb = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")


# ---------------------
# Tokenizer & Model (Base + DPO adapter)
# ---------------------
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"


def load_dpo_model():
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, DPO_ADAPTER_PATH, device_map="auto")
    model.eval()
    model.config.use_cache = True
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    # optional: FlashAttention 2 (if available)
    try:
        model.config.attn_implementation = "flash_attention_2"
    except Exception:
        pass
    return model


model = load_dpo_model()


# ---------------------
# Helpers (same as SFT version)
# ---------------------
def _apply_template(messages_list, tok):
    return [
        tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        for msgs in messages_list
    ]


def _encode_chat_texts(texts, tok, device):
    batch = tok(
        texts,
        return_tensors="pt",
        padding=True,           # enables batching
        truncation=False,
        add_special_tokens=False,
    )
    return {k: v.to(device) for k, v in batch.items()}


def _gen_batch(model, inputs, tok):
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=bool(TEMPERATURE > 0),
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    gens = []
    in_lens = inputs["input_ids"].shape[1]
    for i in range(out.size(0)):
        text = tok.decode(out[i, in_lens:], skip_special_tokens=True).strip()
        gens.append(text)
    return gens


def run_model_on_mtbench_batched(model, tag, mtb, tok, device, batch_size=BATCH_SIZE):
    recs = []
    for i, ex in enumerate(mtb):
        turns = [t for t in ex["prompt"] if isinstance(t, str) and t.strip()]
        recs.append({"i": i, "category": ex.get("category"), "turns": turns})


    # Round 1
    msgs_r1 = [[{"role": "user", "content": r["turns"][0]}] for r in recs]
    texts_r1 = _apply_template(msgs_r1, tok)


    replies1 = [None] * len(recs)
    for start in tqdm(range(0, len(texts_r1), batch_size), desc=f"Generate R1: {tag}"):
        batch_texts = texts_r1[start:start+batch_size]
        inputs = _encode_chat_texts(batch_texts, tok, device)
        gens = _gen_batch(model, inputs, tok)
        replies1[start:start+batch_size] = gens


    # Round 2 (where present)
    idx_second = [idx for idx, r in enumerate(recs) if len(r["turns"]) > 1 and r["turns"][1]]
    msgs_r2, idx_map = [], []
    for idx in idx_second:
        msgs_r2.append([
            {"role": "user", "content": recs[idx]["turns"][0]},
            {"role": "assistant", "content": replies1[idx]},
            {"role": "user", "content": recs[idx]["turns"][1]},
        ])
        idx_map.append(idx)


    replies2 = [None] * len(recs)
    if msgs_r2:
        texts_r2 = _apply_template(msgs_r2, tok)
        for start in tqdm(range(0, len(texts_r2), batch_size), desc=f"Generate R2: {tag}"):
            batch_texts = texts_r2[start:start+batch_size]
            inputs = _encode_chat_texts(batch_texts, tok, device)
            gens = _gen_batch(model, inputs, tok)
            for j, g in enumerate(gens):
                replies2[idx_map[start + j]] = g


    # Pack rows
    rows = []
    for idx, r in enumerate(recs):
        rep = [replies1[idx]]
        if replies2[idx] is not None:
            rep.append(replies2[idx])
        rows.append({
            "index": r["i"],
            "category": r["category"],
            "turns": r["turns"],
            "replies": rep,
            "model_id": tag,
        })
    return rows


# ---------------------
# Run & Save
# ---------------------
rows = run_model_on_mtbench_batched(model, "dpo", mtb, tok, DEVICE, BATCH_SIZE)
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, OUT_FILE)
with open(out_path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")


print(f"Saved DPO generations to {out_path}")
