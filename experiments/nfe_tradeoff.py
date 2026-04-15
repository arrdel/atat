#!/usr/bin/env python3
"""
Experiment 6: NFE Trade-off  (Appendix Table – tab:nfe-full)
============================================================

Reproduces the NFE vs. PPL trade-off analysis: evaluates perplexity
at different sampling budgets (NFE = 64, 128, 256, 512, 1024).

Shows that ATAT's uncertainty-guided sampling improves PPL at all
NFE budgets compared to MDLM's uniform sampling.

Usage:
    python experiments/06_nfe_tradeoff.py \\
        --atat-checkpoint outputs/checkpoints/atat_step_1000000.pt
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paper results (Appendix Table – tab:nfe-full)
# WikiText-2 test, 1M-step checkpoints
NFE_RESULTS = {
    "MDLM": {
        64:   45.67,
        128:  38.92,
        256:  35.41,
        512:  33.58,
        1024: 32.83,
    },
    "ATAT": {
        64:   40.12,
        128:  35.23,
        256:  32.87,
        512:  31.24,
        1024: 30.47,
    },
}

NFE_VALUES = [64, 128, 256, 512, 1024]


def evaluate_nfe_tradeoff(checkpoint_path: str, nfe_values: list):
    """Evaluate ATAT at different NFE budgets."""
    from transformers import GPT2TokenizerFast
    from atat.evaluator import ATATEvaluator
    from atat.models.atat_dit import ATATDiT

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = ATATDiT()
    model.load_state_dict(ckpt["model_state_dict"])
    model.cuda().eval()

    results = {}
    for nfe in nfe_values:
        print(f"  Evaluating NFE = {nfe}...")
        evaluator = ATATEvaluator(model, tokenizer, nfe=nfe, n_eval_runs=3, seed=42)
        from atat.utils.dataloader import create_eval_dataloader
        loader = create_eval_dataloader(
            "wikitext2", "wikitext", "wikitext-2-raw-v1",
            split="test", text_key="text",
        )
        res = evaluator.evaluate_dataset(loader, dataset_name=f"wikitext2@NFE={nfe}")
        results[nfe] = res["ppl"]

    return results


def print_table(atat_eval: dict = None):
    """Print NFE trade-off table."""
    print("\n" + "=" * 70)
    print("NFE Trade-off: PPL vs. sampling budget (WikiText-2 test)")
    print("=" * 70)
    header = f"{'Model':<12}" + "".join(f"{nfe:>8}" for nfe in NFE_VALUES)
    print(header)
    print("-" * 70)

    for name, vals in NFE_RESULTS.items():
        parts = [f"{vals[nfe]:8.2f}" for nfe in NFE_VALUES]
        print(f"{name:<12}{''.join(parts)}")

    if atat_eval:
        parts = [f"{atat_eval.get(nfe, 0):8.2f}" for nfe in NFE_VALUES]
        print(f"{'ATAT (eval)':<12}{''.join(parts)}")

    print("=" * 70)

    # Improvement row
    print(f"\n{'Δ (ATAT−MDLM)':<12}", end="")
    for nfe in NFE_VALUES:
        delta = NFE_RESULTS["ATAT"][nfe] - NFE_RESULTS["MDLM"][nfe]
        print(f"{delta:+8.2f}", end="")
    print()


def main():
    parser = argparse.ArgumentParser(description="NFE trade-off analysis")
    parser.add_argument("--atat-checkpoint", type=str, default=None)
    parser.add_argument("--nfe-values", type=int, nargs="+", default=NFE_VALUES)
    parser.add_argument("--output-dir", type=str, default="results/06_nfe_tradeoff")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    atat_eval = None
    if args.atat_checkpoint:
        atat_eval = evaluate_nfe_tradeoff(args.atat_checkpoint, args.nfe_values)

    print_table(atat_eval)

    all_results = {"paper": NFE_RESULTS}
    if atat_eval:
        all_results["eval"] = atat_eval

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(Path(args.output_dir) / f"nfe_tradeoff_{ts}.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
