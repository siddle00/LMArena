# train_dpo_from_prepared.py
import os, inspect, random, json
import numpy as np
import pandas as pd
import torch
from typing import Optional
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer


# Config (env)
BASE_MODEL     = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
OUT_DIR        = os.environ.get("OUT_DIR", "dpo-mistral7b-from-prepared")
WANDB_PROJECT  = os.environ.get("WANDB_PROJECT", "dpo-prepared")


TRAIN_PATH     = os.environ.get("TRAIN_PATH", "prepared_dataset/train_en_n12000.parquet")
EVAL_PATH      = os.environ.get("EVAL_PATH",  "prepared_dataset/eval_en_n512.parquet") 


N_TRAIN        = int(os.environ.get("N_TRAIN", 10000))
N_EVAL         = int(os.environ.get("N_EVAL", 0))


BATCH_PER_DEV  = int(os.environ.get("BATCH_PER_DEV", 1))
GRAD_ACC       = int(os.environ.get("GRAD_ACC", 8))
LR             = float(os.environ.get("LR", 8e-6))
EVAL_STEPS     = int(os.environ.get("EVAL_STEPS", 200))
SAVE_STEPS     = int(os.environ.get("SAVE_STEPS", 200))
LOGGING_STEPS  = int(os.environ.get("LOGGING_STEPS", 25))
EPOCHS         = float(os.environ.get("EPOCHS", 2.5))
BETA           = float(os.environ.get("BETA", 0.2))


MAX_LEN        = int(os.environ.get("MAX_LEN", 1024))
MAX_PROMPT     = int(os.environ.get("MAX_PROMPT", 300))
MAX_COMP       = int(os.environ.get("MAX_COMP", 1000))


SEED           = int(os.environ.get("SEED", 25))
PRINT_SAMPLES  = int(os.environ.get("PRINT_SAMPLES", 3))


os.environ.setdefault("HF_TORCH_LOAD_ON_GPU", "0")
os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
os.environ.setdefault("WANDB_WATCH", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# Utils
def _is_rank0() -> bool:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0


def _set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def _init_cuda_rank() -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        torch.randn(1, device="cuda")  # warmup
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
        r=32, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )


def _load_df_any(path: str, label: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found at: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    elif ext in [".jsonl", ".json"]:
        df = pd.read_json(path, orient="records", lines=ext==".jsonl")
    else:
        # try parquet then jsonl
        try:
            df = pd.read_parquet(path)
        except Exception:
            df = pd.read_json(path, orient="records", lines=True)
    if _is_rank0():
        print(f"[data] Loaded {label}: {path} ({len(df):,} rows)")
    return df


def _validate_and_trim(df: pd.DataFrame, label: str, n_take: int, seed: int) -> pd.DataFrame:
    required = ["prompt", "chosen", "rejected"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{label} is missing required column: '{c}'. Found: {list(df.columns)}")
    # Basic sanity: non-empty and not identical
    before = len(df)
    df = df[
        df["prompt"].astype(str).str.len().gt(0)
        & df["chosen"].astype(str).str.len().gt(0)
        & df["rejected"].astype(str).str.len().gt(0)
        & (df["chosen"].astype(str).str.strip() != df["rejected"].astype(str).str.strip())
    ].copy()
    if _is_rank0():
        print(f"[data] {label}: kept {len(df):,}/{before:,} after sanity filtering")


    # Optional subsample
    if n_take and n_take > 0 and len(df) > n_take:
        df = df.sample(n=n_take, random_state=seed).reset_index(drop=True)
        if _is_rank0():
            print(f"[data] {label}: subsampled to {len(df):,} rows (n_take={n_take})")
    else:
        df = df.reset_index(drop=True)
    return df


def _to_hf(ds_df: pd.DataFrame) -> HFDataset:
    return HFDataset.from_pandas(ds_df[["prompt", "chosen", "rejected"]], preserve_index=False)


def _show_examples(ds: HFDataset, k: int, title: str):
    if ds is None or len(ds) == 0 or k <= 0: return
    if not _is_rank0(): return
    print(f"\n[{title}] showing {min(k, len(ds))} samples")
    for ex in ds.shuffle(seed=SEED).select(range(min(k, len(ds)))):
        print("="*80)
        print(ex["prompt"][:200].replace("\n", " "))
        print("--- chosen[:160] ---", ex["chosen"][:160].replace("\n", " "))
        print("--- rejected[:160] ---", ex["rejected"][:160].replace("\n", " "))


def _prep_dpo_args(bf16_ok: bool, has_eval: bool) -> DPOConfig:
    save_steps = SAVE_STEPS
    if has_eval and EVAL_STEPS > 0 and (save_steps % EVAL_STEPS != 0):
        save_steps = (save_steps // EVAL_STEPS + 1) * EVAL_STEPS
        if _is_rank0(): print(f"[cfg] Adjusted SAVE_STEPS -> {save_steps} (multiple of EVAL_STEPS={EVAL_STEPS})")


    kwargs = dict(
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
        logging_first_step=True,
        eval_strategy=("steps" if has_eval else "no"),
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        save_safetensors=True,
        load_best_model_at_end=has_eval,               # reload on CPU due to HF_TORCH_LOAD_ON_GPU=0
        metric_for_best_model="eval_rewards/margins",
        greater_is_better=True,
        bf16=bool(bf16_ok),
        fp16=not bf16_ok,
        max_length=MAX_LEN,
        max_prompt_length=MAX_PROMPT,
        max_completion_length=MAX_COMP,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        group_by_length=False,
        length_column_name=None,
        dataloader_num_workers=4,
        report_to=["wandb"],
        seed=SEED,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        tf32=True,
        max_grad_norm=1.0,
    )
    try:
        sig = inspect.signature(DPOConfig.__init__)
        if "beta" in sig.parameters:
            kwargs["beta"] = BETA
    except Exception:
        pass
    return DPOConfig(**kwargs)



def main():
    _set_seed()
    local_rank = _init_cuda_rank()
    if _is_rank0(): print(f"[Rank {local_rank}] starting…")


    # 1) Load prepared splits
    train_df = _load_df_any(TRAIN_PATH, "train split")
    train_df = _validate_and_trim(train_df, "train split", N_TRAIN, SEED)


    eval_df: Optional[pd.DataFrame] = None
    if EVAL_PATH:
        eval_df = _load_df_any(EVAL_PATH, "eval split")
        eval_df = _validate_and_trim(eval_df, "eval split", N_EVAL, SEED)
        if len(eval_df) == 0: eval_df = None


    # 2) Convert to HF Datasets
    train_ds = _to_hf(train_df)
    eval_ds  = _to_hf(eval_df) if eval_df is not None else None


    # 3) Tokenizer, model, LoRA
    tok = _prep_tokenizer()
    model, bf16_ok = _prep_model()
    peft_cfg = _prep_lora()


    # 4) Show a few samples
    _show_examples(train_ds, PRINT_SAMPLES, "Train samples")
    if eval_ds is not None:
        _show_examples(eval_ds, min(2, PRINT_SAMPLES), "Eval samples")


    # 5) Trainer config
    dpo_args = _prep_dpo_args(bf16_ok, has_eval=(eval_ds is not None))


    trainer_kwargs = dict(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_cfg,
    )
    sig = inspect.signature(DPOTrainer.__init__)
    if "tokenizer" in sig.parameters:
        trainer_kwargs["tokenizer"] = tok
    else:
        trainer_kwargs["processing_class"] = tok
    if "precompute_ref_log_probs" in sig.parameters:
        trainer_kwargs["precompute_ref_log_probs"] = True


    trainer = DPOTrainer(**trainer_kwargs)


    if eval_ds is not None:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))


    # 6) Train
    trainer.train()  


    # 7) Save adapter + tokenizer
    if _is_rank0():
        trainer.save_model()
        tok.save_pretrained(OUT_DIR)
        print("Saved LoRA adapter to:", OUT_DIR)


if __name__ == "__main__":
    main()



