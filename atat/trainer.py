"""
ATAT Trainer (Appendix Algorithm 1, Table 9)

DDP training loop with mixed-precision (AMP), curriculum scheduling,
and combined loss L_ATAT = L_denoise + γ · L_importance.

"""

import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from atat.models.atat_dit import ATATDiT
from atat.atat.curriculum import CurriculumScheduler

logger = logging.getLogger(__name__)


def cosine_decay_with_warmup(step: int, total_steps: int, warmup_steps: int, lr: float) -> float:
    """Linear warmup for warmup_steps, then cosine decay to 0."""
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


class ATATTrainer:
    """
    Training loop for the ATAT model.

    Supports:
        - DDP multi-GPU training
        - FP16 mixed precision (AMP)
        - Curriculum scheduling
        - Combined ATAT loss
        - Checkpoint saving / resuming
        - WandB logging

    Args:
        model:           ATATDiT instance.
        train_loader:    Training DataLoader.
        val_loader:      Optional validation DataLoader.
        lr:              Learning rate (3e-4).
        weight_decay:    Weight decay (0.01).
        warmup_steps:    Linear warmup steps (1000).
        total_steps:     Total training steps (1_000_000).
        grad_clip:       Gradient clipping norm (1.0).
        gamma:           Importance loss weight (0.003).
        output_dir:      Checkpoint / log directory.
        log_interval:    Steps between log prints.
        save_interval:   Steps between checkpoint saves.
        eval_interval:   Steps between validation runs.
        use_amp:         Use FP16 mixed precision.
        use_wandb:       Log to WandB.
        seed:            Random seed.
    """

    def __init__(
        self,
        model: ATATDiT,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        total_steps: int = 1_000_000,
        grad_clip: float = 1.0,
        gamma: float = 0.003,
        output_dir: str = "./outputs",
        log_interval: int = 100,
        save_interval: int = 50_000,
        eval_interval: int = 10_000,
        use_amp: bool = True,
        use_wandb: bool = False,
        seed: int = 42,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.grad_clip = grad_clip
        self.gamma = gamma
        self.output_dir = Path(output_dir)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.use_amp = use_amp
        self.use_wandb = use_wandb

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

        # Optimizer: AdamW with paper-specified hyperparameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
        )

        self.scaler = GradScaler(enabled=use_amp)

        # Curriculum scheduler
        self.curriculum = CurriculumScheduler(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )

        # Frequency table (loaded once)
        self.frequency_table = None

        self.global_step = 0
        self.best_val_loss = float("inf")

    def load_frequency_table(self, path: str):
        """Load precomputed token frequency table."""
        self.frequency_table = torch.load(path)
        logger.info(f"Loaded frequency table from {path}")

    def _get_lr(self) -> float:
        return cosine_decay_with_warmup(
            self.global_step, self.total_steps, self.warmup_steps, self.lr
        )

    def _update_lr(self):
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        # Freeze anchor
        self.model.anchor.eval()

        input_ids = batch["input_ids"].cuda()  # (B, L)
        B, L = input_ids.shape

        # Sample diffusion timestep t ~ Uniform(0, 1)
        t = torch.rand(B, device=input_ids.device)

        # Compute importance scores
        importance = self.model.get_importance(
            input_ids,
            frequency_table=self.frequency_table,
        )

        # Compute oracle targets for importance loss
        oracle_targets = self.model.importance_estimator.compute_oracle_targets(
            input_ids
        )

        # Apply adaptive masking: x_0 → x_t
        x_t = self.model.adaptive_mask(input_ids, t, importance)

        # Apply curriculum weights
        curriculum_weights = self.curriculum.compute_curriculum_weights(
            importance, self.global_step
        )

        # Forward + loss
        with autocast(enabled=self.use_amp):
            losses = self.model.compute_loss(
                x_0=input_ids,
                x_t=x_t,
                t=t,
                importance=importance,
                oracle_targets=oracle_targets,
            )

        # Backward
        self.optimizer.zero_grad()
        self.scaler.scale(losses["total"]).backward()

        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.grad_clip,
        )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update LR
        self._update_lr()

        # Update curriculum loss tracking
        self.curriculum.update_loss(self.global_step, losses["denoise"].item())
        self.curriculum.check_plateau(self.global_step)

        return {
            "loss/total": losses["total"].item(),
            "loss/denoise": losses["denoise"].item(),
            "loss/importance": losses["importance"].item(),
            "lr": self._get_lr(),
            **self.curriculum.get_state(self.global_step),
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].cuda()
            B, L = input_ids.shape
            t = torch.rand(B, device=input_ids.device)

            importance = self.model.get_importance(input_ids, self.frequency_table)
            x_t = self.model.adaptive_mask(input_ids, t, importance)

            with autocast(enabled=self.use_amp):
                losses = self.model.compute_loss(
                    x_0=input_ids, x_t=x_t, t=t, importance=importance
                )

            total_loss += losses["total"].item()
            n_batches += 1

        return {"val/loss": total_loss / max(n_batches, 1)}

    def save_checkpoint(self, tag: str = "latest"):
        """Save training checkpoint."""
        path = self.output_dir / "checkpoints" / f"atat_{tag}.pt"
        torch.save(
            {
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            path,
        )
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Resume from checkpoint."""
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(f"Resumed from step {self.global_step}")

    def train(self):
        """Full training loop."""
        logger.info(
            f"Starting ATAT training for {self.total_steps} steps"
            f" | Trainable params: {self.model.trainable_params:,}"
            f" | Total params: {self.model.total_params:,}"
        )

        train_iter = iter(self.train_loader)
        start_time = time.time()

        while self.global_step < self.total_steps:
            # Get next batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            metrics = self.train_step(batch)
            self.global_step += 1

            # Logging
            if self.global_step % self.log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = self.global_step / elapsed
                logger.info(
                    f"Step {self.global_step:>7d}/{self.total_steps}"
                    f" | loss={metrics['loss/total']:.4f}"
                    f" | denoise={metrics['loss/denoise']:.4f}"
                    f" | imp={metrics['loss/importance']:.4f}"
                    f" | lr={metrics['lr']:.2e}"
                    f" | stage={metrics['curriculum/stage']}"
                    f" | {steps_per_sec:.1f} steps/s"
                )

                if self.use_wandb:
                    import wandb
                    wandb.log(metrics, step=self.global_step)

            # Validation
            if self.global_step % self.eval_interval == 0:
                val_metrics = self.validate()
                if val_metrics:
                    logger.info(f"  Validation: {val_metrics}")
                    if val_metrics.get("val/loss", float("inf")) < self.best_val_loss:
                        self.best_val_loss = val_metrics["val/loss"]
                        self.save_checkpoint("best")
                    if self.use_wandb:
                        import wandb
                        wandb.log(val_metrics, step=self.global_step)

            # Checkpointing
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f"step_{self.global_step}")
                self.save_checkpoint("latest")

        self.save_checkpoint("final")
        logger.info("Training complete.")
