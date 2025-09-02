# LMArena
Experiments with Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) on LM-Arena datasets (50k &amp; 140k). Includes dataset prep, EDA, training scripts, evaluation with MT-Bench/LLM-J, and LoRA adapters for Mistral-7B.

# LMarena: SFT & DPO Experiments

Experiments with **Supervised Fine-Tuning (SFT)** and **Direct Preference Optimization (DPO)** on **LM-Arena datasets** (50k & 140k).  
Includes dataset preparation, exploratory analysis, training scripts, evaluation with **MT-Bench** and **LLM-J**, and LoRA adapters for **Mistral-7B-Instruct-v0.2**.

---

## 📂 Repo Structure

```text
LMarena/
├── dataset/          # EDA notebooks, pre-processed parquet (140k, 50k)
├── SFT/              # SFT training script + adapter outputs
├── DPO/              # DPO training scripts + adapters
├── eval/             # MT-Bench generation scripts, judge notebooks, runs
├── WEIGHTS.md        # Links to model artifacts (adapters, merged models)
├── DATA.md           # Links to processed parquet datasets
└── README.md
``` 

### Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/LMarena.git
cd LMarena
````
### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run SFT training

```bash
python SFT/sft_train.py \
  --dataset dataset/normalized_140k.parquet \
  --output_dir SFT/sft_arena55k
```

### 4. Run DPO training

```bash
python DPO/train_dpo.py \
  --dataset dataset/normalized_140k.parquet \
  --output_dir DPO/dpo_arena55k_0830_dulcet_glade_12
```

---

## Evaluation

* **Generation**

```bash
python eval/mtbench_gen.py \
  --model DPO/dpo_arena55k_0830_dulcet_glade_12 \
  --out eval/mtbench_runs/dpo_generations.json
```

* **Judge (LLM-J / Claude)**
  Run notebooks under `eval/`:
* `mtb_response_sft.ipynb`
* `mtb_response_dpo+base.ipynb`
* `eval_llmj_claude.ipynb`

Results (`json`/`jsonl`) are saved in `eval/mtbench_runs/`.

---

## Artifacts

* LoRA adapters for SFT and DPO are hosted separately (see **[WEIGHTS.md](WEIGHTS.md)**).
* Processed parquet datasets are hosted in Hugging Face Datasets (see **[DATA.md](DATA.md)**).
- 📈 **Training & Evaluation Logs (Weights & Biases)**  
  - **SFT**
    - [Arena-55k (Mistral-7B)](https://api.wandb.ai/links/sidarthsrinivasan-ucla/ps585w1s)
    - [Arena-140k (Mistral-7B)](https://api.wandb.ai/links/sidarthsrinivasan-ucla/ps585w1s)
  - **DPO**
    - [Arena-55k (Mistral-7B)](https://api.wandb.ai/links/sidarthsrinivasan-ucla/ps585w1s)
    - [Arena-140k (Mistral-7B)](https://api.wandb.ai/links/sidarthsrinivasan-ucla/ps585w1s)
      
---

## Notes

* Base model: `mistralai/Mistral-7B-Instruct-v0.2`
* Training: LoRA + PEFT
* Datasets: LM-Arena (50k, 140k pre-processed)

---

## License

This repo is for research/educational purposes. Respect the licenses of:

* [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
* [LM-Arena datasets](https://huggingface.co/datasets/lmarena-ai)

---

```
