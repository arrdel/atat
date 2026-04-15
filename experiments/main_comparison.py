#!/usr/bin/env python3
"""
Experiment 1: Main Perplexity Comparison  (Table 1 – tab:main)
==============================================================

Reproduces Table 1 from the paper: perplexity comparison across models.
Baselines: AR (GPT-2 retrained), D3PM, SEDD, MDLM, ADLM.
Ours: ATAT (500K) and ATAT (1M).

Evaluation datasets: LM1B, WikiText-2, LAMBADA, PTB.
All DLM perplexities evaluated at 1000 NFE using NELBO.

Usage:
    python experiments/01_main_comparison.py \\
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

# ── Published baseline results (from respective papers) ──────────────────────

BASELINES = {
    "AR (GPT-2)†": {
        "type": "AR",
        "lm1b": 21.55, "wikitext2": 25.75, "lambada": 51.28, "ptb": 82.05,
        "params": "124M", "train": "124M", "tokens": "65B",
    },
    "D3PM†": {
        "type": "DLM",
        "lm1b": 47.50, "wikitext2": None, "lambada": None, "ptb": None,
        "params": "124M", "train": "124M", "tokens": "33B",
    },
    "SEDD†": {
        "type": "DLM",
        "lm1b": 32.79, "wikitext2": 34.28, "lambada": 49.86, "ptb": 100.09,
        "params": "124M", "train": "124M", "tokens": "33B",
    },
    "MDLM†": {
        "type": "DLM",
        "lm1b": 27.04, "wikitext2": 32.83, "lambada": 47.52, "ptb": 95.26,
        "params": "124M", "train": "124M", "tokens": "33B",
    },
    "ADLM†": {
        "type": "DLM",
        "lm1b": 24.46, "wikitext2": 31.94, "lambada": 44.32, "ptb": 95.37,
        "params": "170M", "train": "170M", "tokens": "65B",
    },
}

DATASETS = ["lm1b", "wikitext2", "lambada", "ptb"]


def evaluate_atat(checkpoint_path: str, datasets: list, nfe: int = 1000):
    """Evaluate ATAT checkpoint using NELBO-based PPL."""
    from transformers import GPT2TokenizerFast
    from atat.evaluator import ATATEvaluator
    from atat.models.atat_dit import ATATDiT

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Load model
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = ATATDiT()
    model.load_state_dict(ckpt["model_state_dict"])
    model.cuda().eval()

    evaluator = ATATEvaluator(model, tokenizer, nfe=nfe, n_eval_runs=3, seed=42)

    results = {}
    for ds_name in datasets:
        from atat.evaluator import BENCHMARKS
        if ds_name in BENCHMARKS:
            cfg = BENCHMARKS[ds_name]
            from atat.utils.dataloader import create_eval_dataloader
            loader = create_eval_dataloader(
                ds_name, cfg["hf_name"], cfg.get("hf_config"),
                split=cfg["split"], text_key=cfg["text_key"],
            )
            res = evaluator.evaluate_dataset(loader, dataset_name=ds_name)
            results[ds_name] = res["ppl"]

    return results


def print_table(atat_results: dict = None):
    """Print Table 1 in formatted ASCII."""
    print("\n" + "=" * 90)
    print("Table 1: Perplexity comparison across models (lower is better)")
    print("†: results from original papers. All DLMs at 1000 NFE.")
    print("=" * 90)
    header = f"{'Type':<5} {'Model':<18} {'LM1B':>8} {'Wiki.':>8} {'LAM.':>8} {'PTB':>8} {'Params':>8} {'Train':>8} {'Tok.':>6}"
    print(header)
    print("-" * 90)

    for name, b in BASELINES.items():
        vals = [f"{b[d]:8.2f}" if b[d] else f"{'--':>8}" for d in DATASETS]
        print(f"{b['type']:<5} {name:<18} {'  '.join(vals)}  {b['params']:>8} {b['train']:>8} {b['tokens']:>6}")

    if atat_results:
        for tag, res in atat_results.items():
            vals = [f"{res.get(d, 0):8.2f}" for d in DATASETS]
            print(f"{'Ours':<5} {tag:<18} {'  '.join(vals)}  {'173M':>8} {'49M':>8} {'65.5B':>6}")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Table 1: Main perplexity comparison")
    parser.add_argument("--atat-checkpoint", type=str, default=None)
    parser.add_argument("--atat-500k-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/01_main_comparison")
    parser.add_argument("--nfe", type=int, default=1000)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    atat_results = {}
    if args.atat_500k_checkpoint:
        atat_results["ATAT (500K)"] = evaluate_atat(args.atat_500k_checkpoint, DATASETS, args.nfe)
    if args.atat_checkpoint:
        atat_results["ATAT (1M)"] = evaluate_atat(args.atat_checkpoint, DATASETS, args.nfe)

    print_table(atat_results)

    # Save
    all_results = {"baselines": BASELINES}
    if atat_results:
        all_results["atat"] = atat_results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(Path(args.output_dir) / f"main_comparison_{ts}.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
