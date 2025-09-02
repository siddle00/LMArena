#!/usr/bin/env python3
import os, json, argparse, random
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser("MT-Bench generation")
    p.add_argument("--base", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    p.add_argument("--adapter", type=str, default=None, help="PEFT adapter dir (DPO). If omitted, runs base.")
    p.add_argument("--tag", type=str, default="dpo", help="model tag to record in output rows")
    p.add_argument("--out_dir", type=str, default="mtbench_runs")
    p.add_argument("--out_file", type=str, default=None, help="optional custom output filename")
    p.add_argument("--seed", type=int, default=25)
    p.add_argument("--max_new_tokens", type=int, default=1000)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--tf32", action="store_true", help="Enable TF32 matmuls on Ampere+ (A100/H100)")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_tokenizer(base_id):
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_model(base_id, adapter_dir, use_bf16=True):
    dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=dtype,
        device_map="auto",           # use all available GPUs
        low_cpu_mem_usage=True,
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(base, adapter_dir, device_map="auto")
    else:
        model = base
    model.eval()
    model.config.use_cache = True
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = model.config.eos_token_id
    return model


def chat_gen(model, tok, device, turns, max_new_tokens, temperature, top_p):
    """
    turns: list of 1-2 user strings
    returns: list of assistant replies
    """
    replies = []
    messages = [{"role": "user", "content": turns[0]}]


    def run_once(msgs):
        prompt_text = tok.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=False,
        )
        batch = tok(
            prompt_text,
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,
        ).to(device)


        with torch.no_grad():
            out = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=bool(temperature > 0),
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )


        gen = tok.decode(out[0][batch["input_ids"].shape[-1]:], skip_special_tokens=True)
        return gen.strip()


    r1 = run_once(messages)
    replies.append(r1)


    if len(turns) > 1 and (turns[1] or "").strip():
        messages.append({"role": "assistant", "content": r1})
        messages.append({"role": "user", "content": turns[1]})
        r2 = run_once(messages)
        replies.append(r2)


    return replies


def main():
    args = parse_args()
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True


    set_seed(args.seed)
    device = get_device()


    # Load prompts
    mtb = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")


    # Load tokenizer/model
    tok = load_tokenizer(args.base)
    model = load_model(args.base, args.adapter)


    rows = []
    for i, ex in enumerate(tqdm(mtb, desc=f"Generating: {args.tag}")):
        turns = [t for t in ex["prompt"] if isinstance(t, str) and t.strip()]
        try:
            replies = chat_gen(
                model, tok, device, turns,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except Exception as e:
            replies = [f"[GENERATION ERROR] {type(e).__name__}: {e}"]


        rows.append({
            "index": i,
            "category": ex.get("category"),
            "turns": turns,
            "replies": replies,
            "model_id": args.tag,
        })


    os.makedirs(args.out_dir, exist_ok=True)
    out_path = args.out_file or f"{args.out_dir}/{args.tag}_generations.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f" Saved {len(rows)} generations to {out_path}")


if __name__ == "__main__":
    main()