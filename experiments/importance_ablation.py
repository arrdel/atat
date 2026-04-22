#!/usr/bin/env python3
"""
Importance Estimation Ablation 
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paper results (Table 2)
ABLATION_RESULTS = {
    "Uniform (baseline)": {"ppl": 42.15, "delta": "—"},
    "Frequency only":     {"ppl": 40.23, "delta": "-1.92"},
    "Contextual only":    {"ppl": 39.81, "delta": "-2.34"},
    "Hybrid (λ=0.7)":    {"ppl": 38.87, "delta": "-3.28"},
}


def evaluate_ablation(checkpoint_dir: str):
    """
    Evaluate importance estimation ablation variants.

    Each variant should have been trained for 50K steps on a 10% OWT subset
    with the corresponding importance_mode setting:
        uniform, frequency_only, learned_only, full
    """
    import torch
    from transformers import GPT2TokenizerFast
    from atat.evaluator import ATATEvaluator
    from atat.models.atat_dit import ATATDiT

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    results = {}

    modes = {
        "Uniform (baseline)": "uniform",
        "Frequency only":     "frequency_only",
        "Contextual only":    "learned_only",
        "Hybrid (λ=0.7)":    "full",
    }

    for label, mode in modes.items():
        ckpt_path = Path(checkpoint_dir) / f"{mode}" / "atat_final.pt"
        if not ckpt_path.exists():
            print(f"  Checkpoint not found: {ckpt_path}, using paper result")
            results[label] = ABLATION_RESULTS[label]["ppl"]
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = ATATDiT(importance_mode=mode)
        model.load_state_dict(ckpt["model_state_dict"])
        model.cuda().eval()

        evaluator = ATATEvaluator(model, tokenizer, nfe=1000, n_eval_runs=3)
        from atat.utils.dataloader import create_eval_dataloader
        loader = create_eval_dataloader(
            "wikitext2", "wikitext", "wikitext-2-raw-v1",
            split="validation", text_key="text",
        )
        res = evaluator.evaluate_dataset(loader, dataset_name="wikitext2")
        results[label] = res["ppl"]

    return results


def print_table(results: dict):
    """Print Table 2."""
    baseline = results.get("Uniform (baseline)", 42.15)
    print("\n" + "=" * 60)
    print("Table 2: Importance estimation ablation")
    print("(WikiText-2 validation PPL, 50K-step runs on 10% OWT)")
    print("=" * 60)
    print(f"{'Variant':<25} {'PPL':>8} {'Δ PPL':>8}")
    print("-" * 60)
    for label, ppl in results.items():
        delta = ppl - baseline
        delta_str = f"{delta:+.2f}" if label != "Uniform (baseline)" else "—"
        marker = " ★" if label == "Hybrid (λ=0.7)" else ""
        print(f"{label:<25} {ppl:8.2f} {delta_str:>8}{marker}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Table 2: Importance estimation ablation")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/02_importance_ablation")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.checkpoint_dir:
        results = evaluate_ablation(args.checkpoint_dir)
    else:
        # Use paper results
        results = {k: v["ppl"] for k, v in ABLATION_RESULTS.items()}
        print("Note: No checkpoint dir provided, showing paper results.")

    print_table(results)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(Path(args.output_dir) / f"importance_ablation_{ts}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
