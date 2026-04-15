# ATAT Experiments

Scripts to reproduce all tables from the paper:
**"Not All Tokens Are Equal: Importance-Aware Masking for Discrete Diffusion Language Models"**.

## Paper → Code Mapping

| Script | Paper Table | Label | Description |
|--------|-------------|-------|-------------|
| `main_comparison.py` | Table 1 | `tab:main` | Perplexity across 7 benchmarks (AR, D3PM, SEDD, MDLM, ADLM, ATAT) |
| `importance_ablation.py` | Table 2 | `tab:hybrid-discovery` | Importance estimation ablation (uniform / frequency / learned / full) |
| `zero_shot.py` | Table 3 | `tab:zero-shot` | Zero-shot perplexity on unseen domains |
| `masking_ablation.py` | Table 4 | `tab:masking-ablation` | Masking strategy ablation (uniform / proportional / inverse / balanced) |
| `lambda_sensitivity.py` | Table 5 | `tab:lambda-sensitivity` | Hybrid weight λ sensitivity (0.0–1.0) |
| `nfe_tradeoff.py` | Appendix | `tab:nfe-full` | NFE vs. PPL trade-off (50–2000 NFE) |
| `pos_importance.py` | Table 6 | `tab:pos-importance` | Mean importance by POS tag |

## Quick Start

```bash
# Display paper results (no checkpoint needed)
python experiments/main_comparison.py
python experiments/zero_shot.py

# Evaluate a trained ATAT checkpoint
python experiments/main_comparison.py --atat-checkpoint outputs/checkpoints/atat_step_1000000.pt
python experiments/zero_shot.py       --atat-checkpoint outputs/checkpoints/atat_step_1000000.pt

# Run ablation experiments (requires ablation checkpoints)
python experiments/importance_ablation.py --checkpoint-dir outputs/ablations/importance/
python experiments/masking_ablation.py    --checkpoint-dir outputs/ablations/masking/
python experiments/lambda_sensitivity.py  --checkpoint-dir outputs/ablations/lambda/

# NFE trade-off
python experiments/nfe_tradeoff.py --atat-checkpoint outputs/checkpoints/atat_step_1000000.pt

# POS analysis (requires spaCy)
pip install spacy && python -m spacy download en_core_web_sm
python experiments/pos_importance.py --atat-checkpoint outputs/checkpoints/atat_step_1000000.pt
```

## Training for Ablations

50K-step ablation runs on 10% OpenWebText subset:

```bash
# Table 2: Importance estimation variants
python atat/scripts/training/train.py \
    --config atat/configs/atat/base.yaml \
    --model.importance.mode uniform --training.max_steps 50000

python atat/scripts/training/train.py \
    --config atat/configs/atat/base.yaml \
    --model.importance.mode frequency_only --training.max_steps 50000

python atat/scripts/training/train.py \
    --config atat/configs/atat/base.yaml \
    --model.importance.mode learned_only --training.max_steps 50000

python atat/scripts/training/train.py \
    --config atat/configs/atat/base.yaml \
    --model.importance.mode full --training.max_steps 50000

# Table 4: Masking strategy variants
python atat/scripts/training/train.py \
    --config atat/configs/atat/base.yaml \
    --masking.strategy proportional --training.max_steps 50000

python atat/scripts/training/train.py \
    --config atat/configs/atat/base.yaml \
    --masking.strategy inverse --training.max_steps 50000

python atat/scripts/training/train.py \
    --config atat/configs/atat/base.yaml \
    --masking.strategy balanced --training.max_steps 50000

# Table 5: Lambda sensitivity
for lambda in 0.0 0.3 0.5 0.7 0.9 1.0; do
    python atat/scripts/training/train.py \
        --config atat/configs/atat/base.yaml \
        --model.importance.lambda_blend $lambda --training.max_steps 50000
done
```

## Full Training

```bash
# Full 1M-step training (Tables 1, 3)
python atat/scripts/training/train.py \
    --config atat/configs/atat/train.yaml
```

## Output Format

Each experiment produces:
1. Formatted ASCII table (console)
2. JSON file with detailed results in `results/<experiment>/`

## Requirements

- PyTorch ≥ 2.0
- transformers ≥ 4.30
- datasets ≥ 2.14
- numpy, tqdm
- spaCy (for experiment 07 only)
