# Not All Tokens Are Equal: Importance-Aware Masking for Discrete Diffusion Language Models

Official implementation of **ATAT (Adaptive Token Attention for Text Diffusion)**.

> **Paper:** *Not All Tokens Are Equal: Importance-Aware Masking for Discrete Diffusion Language Models*  
> **Author:** Adele Chinda  
> **Venue:** NeurIPS 2026 (under review)

---

## Overview

Discrete text diffusion models typically apply **uniform random masking** during the forward process, treating all tokens identically regardless of their linguistic role. ATAT replaces this with **importance-aware adaptive masking**: a lightweight 2-layer MLP (886K parameters) predicts per-token importance scores from a frozen GPT-2 backbone, and the masking scheduler adjusts corruption probabilities based on token difficulty. Combined with a three-stage curriculum and uncertainty-guided sampling, ATAT achieves consistent perplexity improvements across **three** discrete diffusion backbones (MDLM, D3PM, SEDD) at negligible parameter cost.

### Key Contributions

1. **Importance Estimator (§3.2)** — A 2-layer MLP (886K params) operating on frozen GPT-2 hidden states that predicts token-level importance, blending learned attention (λ=0.7) with a log-frequency prior (1−λ=0.3).
2. **Adaptive Masking Scheduler (§3.3)** — Converts importance scores into per-token masking probabilities via time-dependent balanced interpolation: g\_bal(i,t) = t·g\_inv(i) + (1−t)·g\_prop(i), with η=0.3 clipping.
3. **Curriculum Learning (§3.4)** — Three-stage training (Easy 20% → Medium 40% → Hard 40%) that progressively expands the importance range with smooth cosine transitions.
4. **Uncertainty-Guided Sampling (§3.5)** — At inference, prioritizes denoising tokens with highest entropy × importance scores, with κ-sharpened scheduling.
5. **Generalizability (§5)** — ATAT is a model-agnostic plug-in: we demonstrate consistent improvements when applied to D3PM, SEDD, and MDLM backbones.

### Results at a Glance

| Model | Params | WikiText-2 | LAMBADA | PTB | LM1B | AG News | PubMed | ArXiv | Avg |
|-------|--------|-----------|---------|-----|------|---------|--------|-------|-----|
| GPT-2 Small (AR) | 124M | 29.41 | 24.55 | 65.21 | 44.10 | 52.83 | 47.92 | 55.18 | 45.60 |
| D3PM | 169M | 48.72 | 39.88 | 82.45 | 55.31 | 61.28 | 58.14 | 64.73 | 58.64 |
| SEDD | 169M | 44.15 | 35.22 | 76.83 | 51.47 | 57.61 | 54.38 | 60.92 | 54.37 |
| MDLM | 169M | 42.31 | 33.67 | 73.55 | 49.82 | 55.94 | 52.71 | 58.43 | 52.35 |
| **ATAT (ours)** | **173M** | **39.03** | **30.52** | **68.41** | **46.28** | **53.17** | **49.85** | **55.67** | **48.99** |

ATAT achieves a **7.7% relative perplexity reduction** on WikiText-2 over MDLM with only 886K additional trainable parameters (0.5% overhead), training in ~72 hours on 4× RTX 4090 GPUs.

---

## Architecture

ATAT extends MDLM with four modular components (see paper Figure 1):

```
Input tokens x₀
       │
       ▼
┌──────────────────────┐
│  Frozen GPT-2 Small  │  ← 124M params (pretrained, frozen)
│  Anchor Model        │     Produces hidden states h^l
└──────────┬───────────┘
           │ h^l ∈ ℝ^{L×768}
           ▼
┌──────────────────────┐
│  Importance Estimator│  ← 886K params (2-layer MLP)
│  LN→768→256→GELU    │     i^l = λ·σ(MLP(h^l)) + (1-λ)·i^l_freq
│  →256→1→sigmoid      │     λ=0.7, frequency prior blend
└──────────┬───────────┘
           │ importance i ∈ [0,1]^L
           ▼
┌──────────────────────┐
│  Adaptive Masking    │  ← g_bal(i,t) = t·g_inv + (1-t)·g_prop
│  Scheduler           │     η=0.3 clipping, time-dependent
└──────────┬───────────┘
           │ masked tokens x_t
           ▼
┌──────────────────────┐
│  6-Layer Denoiser    │  ← 48M params (init from GPT-2 layers 0-5)
│  Transformer         │     RoPE, pre-LN, adaLN timestep conditioning
│  + Importance proj   │     Input augmented with [i^l; 1-i^l]
└──────────┬───────────┘
           │ logits p_θ(x₀|x_t, t, i)
           ▼
┌──────────────────────┐
│  Uncertainty-Guided  │  ← s^l = entropy(p_θ) × importance(i^l)
│  Sampler             │     κ = 1/(t/T + 0.1) sharpening
└──────────────────────┘
```

**Total parameters:** 173M (124M frozen + 886K estimator + 48M denoiser). **Trainable:** 49M.

---

## Repository Structure

```
.
├── atat/                               # Core ATAT package
│   ├── atat/                           # ATAT modules
│   │   ├── importance_estimator.py     # Token importance MLP (§3.2)
│   │   ├── adaptive_masking.py         # Importance-weighted masking (§3.3)
│   │   ├── curriculum.py               # Curriculum learning scheduler (§3.4)
│   │   └── uncertainty_sampler.py      # Uncertainty-guided sampling (§3.5)
│   ├── models/
│   │   └── atat_dit.py                 # ATATDiT: full ATAT model
│   ├── baselines/                      # Baseline implementations
│   │   ├── ar_transformer/             # Autoregressive GPT baseline
│   │   ├── d3pm/                       # D3PM (Austin et al., 2021)
│   │   ├── mdlm/                       # MDLM (Sahoo et al., 2024)
│   │   └── sedd/                       # SEDD (Lou et al., 2024)
│   ├── experiments/
│   │   └── generalizability/           # §5: Cross-backbone experiments
│   │       ├── atat_plugin.py          # Model-agnostic ATAT wrapper
│   │       ├── train_d3pm_atat.py      # D3PM + ATAT training
│   │       ├── train_sedd_atat.py      # SEDD + ATAT training
│   │       ├── evaluate_generalizability.py
│   │       └── launch.sh              # Pipeline launcher (tmux)
│   ├── configs/                        # Configuration files
│   ├── scripts/                        # Training & evaluation scripts
│   ├── utils/                          # Data loading, visualization
│   ├── trainer.py                      # Training loop
│   └── evaluator.py                    # NELBO-based PPL evaluation
├── paper/                              # LaTeX source (NeurIPS 2026)
│   ├── main.tex                        # Main paper
│   ├── appendix.tex                    # Appendix (A–K)
│   └── figure/                         # Generated figures (PDF/PNG)
├── figures/                            # Architecture diagrams (drawio)
├── results/                            # Evaluation outputs (JSON)
├── setup.py                            # Package installation
├── requirements.txt                    # Dependencies
└── LICENSE                             # Apache 2.0
```

---

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
# Small config for rapid iteration (~5 min on 1 GPU)
python atat/scripts/training/train.py \
  --config atat/configs/atat/debug.yaml
```

### Full Training (Production)

```bash
# Full ATAT training on OpenWebText (1M steps, ~7 days on 4× RTX 4090)
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

## Generalizability Experiments (§5)

ATAT is a model-agnostic plug-in. To reproduce the cross-backbone experiments:

```bash
cd atat

# Launch full pipeline (D3PM baseline → D3PM+ATAT → SEDD baseline → SEDD+ATAT → evaluate)
bash experiments/generalizability/launch.sh

# Or quick test (1K steps)
bash experiments/generalizability/launch.sh --quick
```

---

## Key Hyperparameters

All values match the paper (Appendix A.2, Table 7):

| Symbol | Parameter | Value | Reference |
|--------|-----------|-------|-----------|
| λ | Importance blend weight | 0.7 | §3.2, Eq. 2 |
| β | Importance loss weight | 1.0 | §3.3, Eq. 5 |
| γ | Auxiliary loss coefficient | 0.003 | §3.3, Eq. 5 |
| η | Masking clipping parameter | 0.3 | §3.3, Eq. 3 |
| τ | Oracle temperature | 10 | §3.2, Eq. 4 |
| — | Learning rate | 3×10⁻⁴ | Table 7 |
| — | Weight decay | 0.01 | Table 7 |
| — | Warmup steps | 1,000 | Table 7 |
| — | Max training steps | 1,000,000 | Table 7 |
| — | Batch size | 64 | Table 7 |
| — | Sequence length | 1,024 | Table 7 |
| — | Gradient clipping | 1.0 | Table 7 |

---

## Citation

```bibtex
@inproceedings{chinda2026tokens,
  title={Not All Tokens Are Equal: Importance-Aware Masking for Discrete Diffusion Language Models},
  author={Chinda, Adele},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

If you use the MDLM baseline, please also cite:

```bibtex
@inproceedings{sahoo2024simple,
  title={Simple and Effective Masked Diffusion Language Models},
  author={Sahoo, Subham Sekhar and Arriola, Marianne and Schiff, Yair and Gokaslan, Aaron and Marroquin, Edgar and Kuleshov, Volodymyr},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```

---

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Acknowledgments

Built on the [MDLM](https://github.com/kuleshov-group/mdlm) codebase by Sahoo et al. (NeurIPS 2024). We thank the authors for releasing their code and pretrained models.
