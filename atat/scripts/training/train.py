#!/usr/bin/env python3
"""
ATAT Training Script

Trains the full ATAT model (frozen GPT-2 anchor + importance estimator
+ 6-layer denoiser) following the paper's training configuration.

Training Configuration (Appendix Table 9):
    Optimizer:       AdamW, lr=3e-4, β1=0.9, β2=0.999, ε=1e-8, wd=0.01
    Schedule:        Linear warmup (1K steps) + cosine decay
    Gradient clip:   1.0
    Precision:       FP16 mixed
    Batch size:      64 (16 per GPU × 4 GPUs)
    Sequence length: 1024
    Total steps:     1,000,000 (~72 hours on 4× RTX 4090)

"""

import argparse
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from atat.models.atat_dit import ATATDiT
from atat.trainer import ATATTrainer
from atat.utils.dataloader import create_train_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train ATAT model")

    # Model
    parser.add_argument("--importance-mode", type=str, default="full",
                        choices=["full", "learned_only", "frequency_only", "uniform"])
    parser.add_argument("--lambda-hybrid", type=float, default=0.7)
    parser.add_argument("--gamma", type=float, default=0.003)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--masking-strategy", type=str, default="balanced",
                        choices=["balanced", "proportional", "inverse", "uniform"])

    # Training
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Per-GPU batch size (16 × 4 GPUs = 64 global)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Data
    parser.add_argument("--dataset", type=str, default="openwebtext")
    parser.add_argument("--cache-dir", type=str, default="./data_cache")
    parser.add_argument("--freq-table", type=str, default=None,
                        help="Path to precomputed frequency table")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Limit training documents (for debugging)")

    # Logging / checkpointing
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=50_000)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb-project", type=str, default="atat-diffusion")

    # Resume
    parser.add_argument("--resume", type=str, default=None)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="Disable FP16")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Build model
    logger.info("Building ATAT model...")
    model = ATATDiT(
        importance_mode=args.importance_mode,
        lambda_hybrid=args.lambda_hybrid,
        gamma=args.gamma,
        beta=args.beta,
        eta=args.eta,
    )
    model.masking_scheduler.strategy = args.masking_strategy
    model.cuda()

    logger.info(f"Trainable parameters: {model.trainable_params:,}")
    logger.info(f"Total parameters:     {model.total_params:,}")

    # Build dataloader
    logger.info(f"Loading training data ({args.dataset})...")
    train_loader = create_train_dataloader(
        cache_dir=args.cache_dir,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        max_docs=args.max_docs,
    )

    # WandB
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # Build trainer
    trainer = ATATTrainer(
        model=model,
        train_loader=train_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        total_steps=args.max_steps,
        grad_clip=args.grad_clip,
        gamma=args.gamma,
        output_dir=args.output_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        use_amp=not args.no_amp,
        use_wandb=args.wandb,
        seed=args.seed,
    )

    # Load frequency table
    if args.freq_table:
        trainer.load_frequency_table(args.freq_table)

    # Resume
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
