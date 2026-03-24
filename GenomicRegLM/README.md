# GenomicRegLM

A standalone, extracted codebase for **multimodal gene expression prediction** from:
- DNA sequence
- ATAC-seq accessibility signals
- ChIP-seq chromatin marks
- genomic coordinates and sample metadata

This folder is intended to be copied out as a new repository.

---

## 1. What is extracted here

This directory contains only the new functionality developed for the new architecture, separated from the legacy PDLLMs repository:

- `src/genomicreglm/`:
  - config loader
  - tokenizer(s)
  - multimodal dataset contracts
  - collator
  - synthetic data builder for smoke training
  - model factory + baseline multimodal model skeleton
  - lightweight trainer
  - CLI training entrypoint
- `configs/`:
  - data/model/train/experiment templates
- `docs/zh/`:
  - Chinese quickstart
  - architecture roadmap
  - experiment matrix

---

## 2. Recommended new-repo structure

```text
GenomicRegLM/
├─ README.md
├─ pyproject.toml
├─ configs/
│  ├─ data/
│  ├─ model/
│  ├─ train/
│  └─ experiment/
├─ docs/
│  └─ zh/
└─ src/
   └─ genomicreglm/
```

---

## 3. Quick start

### 3.1 Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch pyyaml
```

### 3.2 Run synthetic end-to-end smoke training

```bash
PYTHONPATH=src python -m genomicreglm.cli.train \
  --data-config configs/data/maize_multiomics.yaml \
  --model-config configs/model/transformer_multimodal.yaml \
  --train-config configs/train/finetune_expression_regression.yaml \
  --synthetic \
  --num-samples 64
```

You should see epoch losses and a final summary line like:

```text
[GenomicRegLM] done | data=... | model=...
```

---

## 4. Current capability boundary (important)

### Done
- Config-driven training entrypoint (`data/model/train` YAML)
- Synthetic multimodal samples for smoke testing
- Tokenizer/model/trainer wiring and runnable loop

### TODO (next priorities)
- Real-data builder for FASTA + GTF + bigWig + expression matrix
- Proper sequence encoder backend selection (Transformer/Mamba implementations)
- Evaluation metrics module (Pearson/Spearman/MAE/MSE) in pipeline
- Checkpointing / logging / reproducibility utilities

---

## 5. Config guide

- `configs/data/maize_multiomics.yaml`
  - defines reference genome, coordinate window, tracks, split strategy
- `configs/model/transformer_multimodal.yaml`
  - baseline Transformer route
- `configs/model/mamba_multimodal.yaml`
  - Mamba route placeholder
- `configs/train/finetune_expression_regression.yaml`
  - optimizer, scheduler, precision, epochs
- `configs/experiment/exp_matrix_v1.yaml`
  - first ablation plan across model/tokenizer/fusion/split

---

## 6. How to create a separate Git repository

Inside current repo root:

```bash
mkdir -p ../GenomicRegLM_repo
rsync -av GenomicRegLM/ ../GenomicRegLM_repo/
cd ../GenomicRegLM_repo
git init
git add .
git commit -m "init: extracted GenomicRegLM standalone repository"
```

---

## 7. Chinese docs

- `docs/zh/quickstart_zh.md`
- `docs/zh/roadmap_zh.md`
- `docs/zh/experiment_matrix_v1_zh.md`

