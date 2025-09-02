# train_sft.py
import os, json, random, inspect, torch
import numpy as np
import pandas as pd
from typing import Optional
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model


BASE_MODEL     = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
OUT_DIR        = os.environ.get("OUT_DIR", "sft-mistral7b-arena140k")
WANDB_PROJECT  = os.environ.get("WANDB_PROJECT", "sft-arena-quickrun")


TRAIN_PATH     = os.environ.get("TRAIN_PATH", "prepared_sft_dataset/train_sft_en_n12000.parquet")
EVAL_PATH      = os.environ.get("EVAL_PATH",  "prepared_sft_dataset/eval_sft_en_n512.parquet") 

N_TRAIN        = int(os.environ.get("N_TRAIN", 10000))
N_EVAL         = int(os.environ.get("N_EVAL", 512))


MAX_LEN        = int(os.environ.get("MAX_LEN", 1024))
MAX_PROMPT     = int(os.environ.get("MAX_PROMPT", 300))
MAX_COMP       = int(os.environ.get("MAX_COMP", 1000))


BATCH_PER_DEV  = int(os.environ.get("BATCH_PER_DEV", 1))
GRAD_ACC       = int(os.environ.get("GRAD_ACC", 8))
LR             = float(os.environ.get("LR", 5e-5))
EVAL_STEPS     = int(os.environ.get("EVAL_STEPS", 200))
SAVE_STEPS     = int(os.environ.get("SAVE_STEPS", 200))
LOGGING_STEPS  = int(os.environ.get("LOGGING_STEPS", 25))
EPOCHS         = float(os.environ.get("EPOCHS", 2.0))


SEED           = int(os.environ.get("SEED", 25))
PRINT_SAMPLES  = int(os.environ.get("PRINT_SAMPLES", 3))


os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
os.environ.setdefault("WANDB_WATCH", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _init_cuda_rank():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        torch.randn(1, device="cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return local_rank


def _prep_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left" 
    return tok


def _prep_model():
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if bf16_ok else torch.float16,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        try:
            model.get_input_embeddings().requires_grad_(True)
        except Exception:
            pass
    return model, bf16_ok


def _prep_lora():
    return LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )


# Load prepared SFT splits 
def _load_df_any(path: str, label: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found at: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    elif ext in [".jsonl", ".json"]:
        df = pd.read_json(path, orient="records", lines=(ext == ".jsonl"))
    else:
        try:
            df = pd.read_parquet(path)
        except Exception:
            df = pd.read_json(path, orient="records", lines=True)
    print(f"[data] Loaded {label}: {path} ({len(df):,} rows)")
    return df


def _validate_and_trim_sft(df: pd.DataFrame, label: str, n_take: int, seed: int) -> pd.DataFrame:
    required = ["prompt", "completion"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{label} missing required column '{c}'. Found: {list(df.columns)}")
    df = df[required].copy().dropna()
    for c in required:
        df[c] = df[c].astype(str)
    before = len(df)
    df = df[
        df["prompt"].str.strip().str.len().gt(0) &
        df["completion"].str.strip().str.len().gt(0)
    ].copy()
    df = df.drop_duplicates(subset=["prompt","completion"]).reset_index(drop=True)
    print(f"[data] {label}: kept {len(df):,}/{before:,} after sanity + dedupe")
    if n_take and n_take > 0 and len(df) > n_take:
        df = df.sample(n=n_take, random_state=seed).reset_index(drop=True)
        print(f"[data] {label}: subsampled to {len(df):,} rows (n_take={n_take})")
    return df


def _to_hf_sft(df: pd.DataFrame) -> HFDataset:
    return HFDataset.from_pandas(df[["prompt","completion"]], preserve_index=False)


#Tokenization & Collator
def _tokenize_for_sft(train_ds, eval_ds, tok):
    """
    {prompt, completion} -> {input_ids, attention_mask, labels}
    Labels are -100 over prompt tokens; completion tokens are supervised.
    """
    eos = tok.eos_token or ""


    def build_ids(prompt_text, completion_text):
        if eos and not completion_text.endswith(eos):
            completion_text = completion_text + eos
        p_ids = tok(prompt_text, add_special_tokens=False).input_ids
        c_ids = tok(completion_text, add_special_tokens=False).input_ids
        if MAX_PROMPT > 0 and len(p_ids) > MAX_PROMPT:
            p_ids = p_ids[-MAX_PROMPT:]
        if MAX_COMP > 0 and len(c_ids) > MAX_COMP:
            c_ids = c_ids[:MAX_COMP]
        total = len(p_ids) + len(c_ids)
        if MAX_LEN > 0 and total > MAX_LEN:
            overflow = total - MAX_LEN
            if overflow < len(p_ids):
                p_ids = p_ids[overflow:]
            else:
                p_keep = max(1, len(p_ids) - overflow)
                p_ids = p_ids[-p_keep:]
        input_ids = p_ids + c_ids
        labels = [-100] * len(p_ids) + c_ids
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


    def tok_map(batch):
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for p, c in zip(batch["prompt"], batch["completion"]):
            pack = build_ids(p, c)
            out["input_ids"].append(pack["input_ids"])
            out["attention_mask"].append(pack["attention_mask"])
            out["labels"].append(pack["labels"])
        return out


    cols_to_remove = train_ds.column_names
    train_tok = train_ds.map(tok_map, batched=True, remove_columns=cols_to_remove)
    eval_tok  = eval_ds.map(tok_map, batched=True, remove_columns=eval_ds.column_names) if eval_ds is not None else None
    return train_tok, eval_tok


class _CausalCollator:
    """Pads input_ids/attention_mask with tokenizer.pad_token_id / 0 and labels with -100."""
    def __init__(self, tok):
        self.tok = tok
        self.pad_id = tok.pad_token_id
    def __call__(self, batch):
        side = self.tok.padding_side
        max_len = max(len(x["input_ids"]) for x in batch)
        def pad(seq, pad_val, side):
            pad_len = max_len - len(seq)
            if pad_len <= 0: return seq
            pad_chunk = [pad_val] * pad_len
            return (pad_chunk + seq) if side == "left" else (seq + pad_chunk)
        input_ids = torch.tensor([pad(x["input_ids"], self.pad_id, side) for x in batch], dtype=torch.long)
        attention = torch.tensor([pad(x["attention_mask"], 0, side) for x in batch], dtype=torch.long)
        labels    = torch.tensor([pad(x["labels"], -100, side) for x in batch], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def _show_examples(ds, k=2, title="samples"):
    if ds is None or len(ds) == 0 or k <= 0: return
    print(f"\n{title}:")
    for ex in ds.shuffle(seed=SEED).select(range(min(k, len(ds)))):
        print("="*80)
        print(ex["prompt"])
        print("--- completion (first 400 chars) ---")
        print(ex["completion"][:400])


def _prep_training_args(bf16_ok, has_eval):
    save_steps = SAVE_STEPS
    if has_eval and EVAL_STEPS > 0 and (save_steps % EVAL_STEPS != 0):
        save_steps = (save_steps // EVAL_STEPS + 1) * EVAL_STEPS
        print(f"[cfg] Adjusted SAVE_STEPS -> {save_steps} to be a multiple of EVAL_STEPS={EVAL_STEPS}")
    return TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_PER_DEV,
        per_device_eval_batch_size=max(1, BATCH_PER_DEV),
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.05,
        logging_steps=LOGGING_STEPS,
        eval_strategy=("steps" if has_eval else "no"),
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bool(bf16_ok),
        fp16=not bf16_ok,
        gradient_checkpointing=True,
        optim="adamw_torch",
        group_by_length=False,
        report_to=["wandb"],
        seed=SEED,
        ddp_find_unused_parameters=False,
        tf32=True,
        max_grad_norm=1.0,
    )


def main():
    _set_seed(SEED)
    local_rank = _init_cuda_rank()
    print(f"[Rank {local_rank}] starting…")


    tok = _prep_tokenizer()
    model, bf16_ok = _prep_model()
    peft_cfg = _prep_lora()
    model = get_peft_model(model, peft_cfg)

    train_df = _load_df_any(TRAIN_PATH, "train split (SFT)")
    train_df = _validate_and_trim_sft(train_df, "train split (SFT)", N_TRAIN, SEED)


    eval_df: Optional[pd.DataFrame] = None
    if EVAL_PATH:
        eval_df = _load_df_any(EVAL_PATH, "eval split (SFT)")
        eval_df = _validate_and_trim_sft(eval_df, "eval split (SFT)", N_EVAL, SEED)
        if len(eval_df) == 0:
            eval_df = None


    raw_train = _to_hf_sft(train_df)
    raw_eval  = _to_hf_sft(eval_df) if eval_df is not None else None


    if PRINT_SAMPLES > 0 and local_rank == 0:
        _show_examples(raw_train, PRINT_SAMPLES, title="Train samples")
        if raw_eval is not None:
            _show_examples(raw_eval, min(PRINT_SAMPLES, 2), title="Eval samples")


    # Tokenize
    train_ds, eval_ds = _tokenize_for_sft(raw_train, raw_eval, tok)
    collator = _CausalCollator(tok)


    args = _prep_training_args(bf16_ok, has_eval=(eval_ds is not None))


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tok,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if eval_ds is not None else None,
    )


    trainer.train()


    if int(os.environ.get("RANK", 0)) == 0:
        trainer.save_model()
        tok.save_pretrained(OUT_DIR)
        print("Saved SFT LoRA adapter to:", OUT_DIR)


if __name__ == "__main__":
    main()




