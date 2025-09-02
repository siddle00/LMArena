# train_dpo.py
import os, inspect, random, json, torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer


BASE_MODEL     = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
OUT_DIR        = os.environ.get("OUT_DIR", "dpo-mistral7b-arena55k")
WANDB_PROJECT  = os.environ.get("WANDB_PROJECT", "dpo-arena-quickrun")


N_TRAIN        = int(os.environ.get("N_TRAIN", 10000))  
N_EVAL         = int(os.environ.get("N_EVAL", 512))     
MAX_LEN        = int(os.environ.get("MAX_LEN", 1024))
MAX_PROMPT     = int(os.environ.get("MAX_PROMPT", 300))
MAX_COMP       = int(os.environ.get("MAX_COMP", 1000))


BATCH_PER_DEV  = int(os.environ.get("BATCH_PER_DEV", 1))
GRAD_ACC       = int(os.environ.get("GRAD_ACC", 8))
LR             = float(os.environ.get("LR", 8e-6))
EVAL_STEPS     = int(os.environ.get("EVAL_STEPS", 200))
SAVE_STEPS     = int(os.environ.get("SAVE_STEPS", 200))  
LOGGING_STEPS  = int(os.environ.get("LOGGING_STEPS", 25))
EPOCHS         = float(os.environ.get("EPOCHS", 2.5))
BETA           = float(os.environ.get("BETA", 0.2))


SEED           = int(os.environ.get("SEED", 25))
PRINT_SAMPLES  = int(os.environ.get("PRINT_SAMPLES", 3)) 


# Defaults
os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
os.environ.setdefault("WANDB_WATCH", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _init_cuda_rank():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
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
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj"
        ]
    )


# Prep dataset
def _as_text(x):
    """
    Arena fields are sometimes lists or JSON-encoded lists; return plain text.
    """
    if isinstance(x, list):
        return "\n".join(s for s in x if isinstance(s, str)).strip()
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return "\n".join(str(t) for t in arr).strip()
            except Exception:
                pass
        return s
    return "" if x is None else str(x)


def _prep_datasets():
    """
    Normalize Arena-55k into (prompt, chosen, rejected) with correct response text and winner flags.
    Dataset features:
      ['id','model_a','model_b','prompt','response_a','response_b','winner_model_a','winner_model_b','winner_tie']
    """
    def make_prompt(p: str) -> str:
        return f"### Instruction:\n{_as_text(p)}\n\n### Response:\n"


    def normalize(ex):
        # Prompt
        p = ex.get("prompt", "")
        # Response texts (NOT model names)
        a_text = _as_text(ex.get("response_a") or "")
        b_text = _as_text(ex.get("response_b") or "")
        # Winner flags -> booleans
        wa = ex.get("winner_model_a", 0) or 0
        wb = ex.get("winner_model_b", 0) or 0
        wt = ex.get("winner_tie",     0) or 0
        try: wa = bool(int(wa))
        except Exception: wa = bool(wa)
        try: wb = bool(int(wb))
        except Exception: wb = bool(wb)
        try: wt = bool(int(wt))
        except Exception: wt = bool(wt)


        if wa and not wb and not wt:
            chosen, rejected = a_text, b_text
        elif wb and not wa and not wt:
            chosen, rejected = b_text, a_text
        else:
            # tie/inconsistent -> drop via ok()
            chosen = rejected = ""


        return {"prompt": make_prompt(p), "chosen": chosen, "rejected": rejected}


    def ok(ex):
        # Ensure real instruction body + non-empty, non-identical responses
        before_resp = ex["prompt"].split("### Response:")[0]
        core = before_resp.replace("### Instruction:", "", 1).strip()
        return bool(core and ex["chosen"].strip() and ex["rejected"].strip() and (ex["chosen"].strip() != ex["rejected"].strip()))


    ds = load_dataset("lmarena-ai/arena-human-preference-55k")
    # Only 'train' exists; map + filter it
    cleaned = ds["train"].map(normalize, remove_columns=ds["train"].column_names).filter(ok)


    # Create eval split from cleaned data
    split = cleaned.train_test_split(test_size=0.02, seed=SEED)


    train_ds = split["train"].select(range(min(N_TRAIN, len(split["train"])))) if N_TRAIN > 0 else split["train"]
    eval_ds  = split["test"].select(range(min(N_EVAL,  len(split["test"]))))  if N_EVAL  > 0 else None
    return train_ds, eval_ds


def _show_examples(ds, k=2, title="samples"):
    if ds is None or len(ds) == 0 or k <= 0:
        return
    print(f"\n{title}:")
    for ex in ds.shuffle(seed=SEED).select(range(min(k, len(ds)))):
        print("="*80)
        print(ex["prompt"])
        print("--- chosen (first 400 chars) ---")
        print(ex["chosen"][:400])
        print("--- rejected (first 400 chars) ---")
        print(ex["rejected"][:400])


def _prep_dpo_args(bf16_ok, has_eval):
    # Ensure save/eval compatibility for load_best_model_at_end
    save_steps = SAVE_STEPS
    if has_eval and EVAL_STEPS > 0 and (save_steps % EVAL_STEPS != 0):
        save_steps = (save_steps // EVAL_STEPS + 1) * EVAL_STEPS
        print(f"[cfg] Adjusted SAVE_STEPS -> {save_steps} to be a multiple of EVAL_STEPS={EVAL_STEPS}")


    dpo_kwargs = dict(
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
        save_steps=EVAL_STEPS,
        save_total_limit=2,
        load_best_model_at_end=has_eval,
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
            dpo_kwargs["beta"] = BETA
    except (ValueError, TypeError):
        pass


    return DPOConfig(**dpo_kwargs)


def main():
    _set_seed(SEED)
    local_rank = _init_cuda_rank()
    print(f"[Rank {local_rank}] starting…")


    tok = _prep_tokenizer()
    model, bf16_ok = _prep_model()
    peft_cfg = _prep_lora()
    train_ds, eval_ds = _prep_datasets()

    if PRINT_SAMPLES > 0 and local_rank == 0:
        _show_examples(train_ds, PRINT_SAMPLES, title="Train samples")
        if eval_ds is not None:
            _show_examples(eval_ds, min(PRINT_SAMPLES, 2), title="Eval samples")


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

    trainer.train()

    if int(os.environ.get("RANK", 0)) == 0:
        trainer.save_model()
        tok.save_pretrained(OUT_DIR)
        print("Saved LoRA adapter to:", OUT_DIR)

if __name__ == "__main__":
    main()
