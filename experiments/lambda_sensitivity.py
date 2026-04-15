#!/usr/bin/env python3
"""
Experiment 5: Lambda Sensitivity  (Table 5 – tab:lambda-sensitivity)
====================================================================

Reproduces Table 5 from the paper: sensitivity to the hybrid weight λ
that balances contextual and frequency importance.

    i^l = λ · i^l_context + (1 - λ) · i^l_freq

50K-step ablation runs on 10% OWT, WikiText-2 validation PPL.

Usage:
    python experiments/05_lambda_sensitivity.py \\
        --checkpoint-dir outputs/ablations/lambda/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paper results (Table 5)
LAMBDA_RESULTS = {
    0.0: 40.23,   # frequency only
    0.3: 39.54,
    0.5: 39.31,
    0.7: 38.87,   # optimal
    0.9: 39.71,
    1.0: 39.81,   # contextual only
}


def evaluate_lambda(checkpoint_dir: str):
    """Evaluate λ sensitivity from trained checkpoints."""
    import torch
    from transformers import GPT2TokenizerFast
    from atat.evaluator import ATATEvaluator
    from atat.models.atat_dit import ATATDiT

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    results = {}

    for lam in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
        tag = f"lambda_{lam:.1f}"
        ckpt_path = Path(checkpoint_dir) / tag / "atat_final.pt"
        if not ckpt_path.exists():
            print(f"  Checkpoint not found: {ckpt_path}, using paper result")
            results[lam] = LAMBDA_RESULTS[lam]
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = ATATDiT(lambda_hybrid=lam)
        model.load_state_dict(ckpt["model_state_dict"])
        model.cuda().eval()

        evaluator = ATATEvaluator(model, tokenizer, nfe=1000, n_eval_runs=3)
        from atat.utils.dataloader import create_eval_dataloader
        loader = create_eval_dataloader(
            "wikitext2", "wikitext", "wikitext-2-raw-v1",
            split="validation", text_key="text",
        )
        res = evaluator.evaluate_dataset(loader, dataset_name="wikitext2")
        results[lam] = res["ppl"]

    return results


def print_table(results: dict):
    """Print Table 5."""
    best_lam = min(results, key=results.get)
    print("\n" + "=" * 50)
    print("Table 5: Sensitivity to hybrid weight λ")
    print("(50K-step ablation, WikiText-2 val PPL)")
    print("=" * 50)
    print(f"{'λ':>6} {'PPL':>8}  {'Note'}")
    print("-" * 50)
    for lam in sorted(results):
        ppl = results[lam]
        note = ""
        if lam == 0.0:
            note = "frequency only"
        elif lam == 1.0:
            note = "contextual only"
        elif lam == best_lam:
            note = "★ optimal"
        print(f"{lam:6.1f} {ppl:8.2f}  {note}")
    print("-" * 50)
    span = max(results.values()) - min(results.values())
    print(f"Range: {span:.2f} PPL (robust across [0.5, 0.9])")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Table 5: Lambda sensitivity")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/05_lambda_sensitivity")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.checkpoint_dir:
        results = evaluate_lambda(args.checkpoint_dir)
    else:
        results = dict(LAMBDA_RESULTS)
        print("Note: No checkpoint dir provided, showing paper results.")

    print_table(results)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(Path(args.output_dir) / f"lambda_sensitivity_{ts}.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)


if __name__ == "__main__":
    main()
