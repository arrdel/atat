# MDLM Baseline

Standard MDLM (Masked Diffusion Language Model) with **uniform random masking** — the primary baseline for comparison with ATAT's adaptive masking.

> Sahoo et al., "Simple and Effective Masked Diffusion Language Models," NeurIPS 2024.

## Architecture

Matches ATAT for fair comparison:

| Component | Value |
|-----------|-------|
| Backbone | DiT (Diffusion Transformer) |
| Hidden size | 768 |
| Layers | 12 |
| Heads | 12 |
| Parameters | ~169M |
| Masking | Uniform random |
| Noise schedule | Log-linear |

## Usage

```bash
# Quick test
python train_mdlm_baseline.py --max-steps 100 --num-gpus 1

# Full training (matches paper Table 1)
python train_mdlm_baseline.py --max-steps 1000000 --num-gpus 4
```

## Notes

This baseline uses uniform Bernoulli masking with no importance weighting, curriculum learning, or adaptive scheduling. All other training settings (optimizer, LR, batch size, data) match ATAT for fair comparison.
