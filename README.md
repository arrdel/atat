# Not All Tokens Are Equal: Importance-Aware Masking for Discrete Diffusion Language Models

Official implementation of **ATAT (Adaptive Token Attention for Text Diffusion)**.

---

## Overview

Discrete text diffusion models typically apply **uniform random masking** during the forward process, treating all tokens identically regardless of their linguistic role. ATAT replaces this with **importance-aware adaptive masking**: a lightweight 2-layer MLP (886K parameters) predicts per-token importance scores from a frozen GPT-2 backbone, and the masking scheduler adjusts corruption probabilities based on token difficulty. Combined with a three-stage curriculum and uncertainty-guided sampling, ATAT achieves consistent perplexity improvements across **three** discrete diffusion backbones (MDLM, D3PM, SEDD) at negligible parameter cost.

### Key Contributions

1. **Importance Estimator** — A 2-layer MLP (886K params) operating on frozen GPT-2 hidden states that predicts token-level importance, blending learned attention (λ=0.7) with a log-frequency prior (1−λ=0.3).
2. **Adaptive Masking Scheduler** — Converts importance scores into per-token masking probabilities via time-dependent balanced interpolation: g\_bal(i,t) = t·g\_inv(i) + (1−t)·g\_prop(i), with η=0.3 clipping.
3. **Curriculum Learning** — Three-stage training (Easy 20% → Medium 40% → Hard 40%) that progressively expands the importance range with smooth cosine transitions.
4. **Uncertainty-Guided Sampling** — At inference, prioritizes denoising tokens with highest entropy × importance scores, with κ-sharpened scheduling.
5. **Generalizability** — ATAT is a model-agnostic plug-in: we demonstrate consistent improvements when applied to D3PM, SEDD, and MDLM backbones.

## Installation

### Prerequisites

- Python ≥ 3.9
- CUDA ≥ 12.1
- 4× NVIDIA GPU with ≥ 24 GB VRAM (RTX 4090, A5000, or A100)

### Setup

```bash
# Clone the repository
git clone https://github.com/arrdel/atat.git
cd atat

# Create conda environment
conda create -n atat python=3.9 -y
conda activate atat

# Install PyTorch with CUDA 12.1
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Install ATAT in development mode
pip install -e .

# (Optional) Flash Attention for faster training
pip install flash-attn==2.8.3 --no-build-isolation
```

### Data Preparation

```bash
# Download and cache all datasets
python atat/scripts/data/download_datasets.py

# Precompute token frequency table (for importance estimator)
python -m atat.utils.frequency --data-dir ./data_cache --output freq_table.pt
```

---

## Training

### Quick Start (Debug)

```bash
# Small config for rapid iteration
python atat/scripts/training/train.py \
  --config atat/configs/atat/debug.yaml
```

### Full Training (Production)

```bash
# Full ATAT training on OpenWebText 
python atat/scripts/training/train.py \
  --config atat/configs/atat/train.yaml
```

### Training with Custom Hyperparameters

```bash
# Override any config value via CLI
python atat/scripts/training/train.py \
  --config atat/configs/atat/base.yaml \
  --training.lr 1e-4 \
  --training.max_steps 500000 \
  --data.batch_size 32
```

---

## Evaluation

### Perplexity Evaluation (NELBO-based)

```bash
# Evaluate on all 7 benchmarks (as in Table 1)
python atat/scripts/evaluation/evaluate_ppl.py \
  --checkpoint <path_to_checkpoint> \
  --benchmarks wikitext2 lambada ptb lm1b ag_news pubmed arxiv \
  --nfe 1000 \
  --num-runs 3

# Quick single-benchmark evaluation
python atat/scripts/evaluation/evaluate_ppl.py \
  --checkpoint <path_to_checkpoint> \
  --benchmarks wikitext2 \
  --nfe 1000
```

---

## Generalizability Experiments

ATAT is a model-agnostic plug-in. To reproduce the cross-backbone experiments:

```bash
cd atat

# Launch full pipeline (D3PM baseline → D3PM+ATAT → SEDD baseline → SEDD+ATAT → evaluate)
bash experiments/generalizability/launch.sh

# Or quick test (1K steps)
bash experiments/generalizability/launch.sh --quick
```

---

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Acknowledgments

Built on the [MDLM](https://github.com/kuleshov-group/mdlm) codebase by Sahoo et al. (NeurIPS 2024). We thank the authors for releasing their code and pretrained models.
