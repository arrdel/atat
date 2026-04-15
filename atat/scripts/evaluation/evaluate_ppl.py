#!/usr/bin/env python3
"""
ATAT Perplexity Evaluation Script

Evaluates ATAT and baseline models using NELBO-based perplexity
on all 7 benchmarks from the paper.

Protocol (Appendix §D.2):
    - NELBO with 1000 NFE
    - Log-linear noise schedule: α_t = 1 - t
    - Seed 42, averaged over 3 evaluation runs
    - Benchmarks: WikiText-2, LAMBADA, PTB, LM1B, AG News, PubMed, ArXiv

Usage:
    python scripts/evaluation/evaluate_ppl.py \\
        --checkpoint outputs/checkpoints/atat_step_1000000.pt \\
        --benchmarks wikitext2 lambada ptb \\
        --nfe 1000 \\
        --output-dir results/evaluation/
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import GPT2TokenizerFast

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from atat.evaluator import ATATEvaluator, BENCHMARKS
from atat.models.atat_dit import ATATDiT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ATAT perplexity (NELBO-based)"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to ATAT checkpoint",
    )
    parser.add_argument(
        "--benchmarks", type=str, nargs="+",
        default=list(BENCHMARKS.keys()),
        help="Benchmarks to evaluate (default: all 7)",
    )
    parser.add_argument("--nfe", type=int, default=1000, help="Number of function evaluations")
    parser.add_argument("--n-runs", type=int, default=3, help="Number of evaluation runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--cache-dir", type=str, default="./data_cache")
    parser.add_argument("--output-dir", type=str, default="results/evaluation/")
    parser.add_argument("--freq-table", type=str, default=None, help="Frequency table path")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = ATATDiT()
    model.load_state_dict(ckpt["model_state_dict"])
    model.cuda().eval()

    logger.info(f"Trainable params: {model.trainable_params:,}")
    logger.info(f"Total params: {model.total_params:,}")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Load frequency table if provided
    freq_table = None
    if args.freq_table:
        freq_table = torch.load(args.freq_table).cuda()

    # Evaluate
    evaluator = ATATEvaluator(
        model=model,
        tokenizer=tokenizer,
        nfe=args.nfe,
        seed=args.seed,
        n_eval_runs=args.n_runs,
    )

    results = evaluator.evaluate_all_benchmarks(
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        frequency_table=freq_table,
    )

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_step = ckpt.get("global_step", "unknown")
    output_path = Path(args.output_dir) / f"ppl_step{global_step}_{ts}.json"

    output = {
        "checkpoint": args.checkpoint,
        "global_step": global_step,
        "nfe": args.nfe,
        "n_runs": args.n_runs,
        "seed": args.seed,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
