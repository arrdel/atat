"""
Curriculum Learning Scheduler (§3.3)

Implements the three-stage macro-level training curriculum:

    Stage 1 – Easy   (first 20% of steps):  focus on tokens with i ∈ [0, 0.3]
    Stage 2 – Medium (next  40% of steps):  focus on tokens with i ∈ [0.3, 0.7]
    Stage 3 – Hard   (final 40% of steps):  full range i ∈ [0, 1.0]

Dynamic adaptation: extends a stage if denoising loss decreases < 1% per 10K steps.

Training config (Table 9):
    Easy phase   → Steps 0–200K    (first 20%)
    Medium phase → Steps 200K–600K (next 40%)
    Hard phase   → Steps 600K–1M   (final 40%)
    Phase transition: smooth cosine interpolation.
"""

import math
from typing import Dict, Optional, Tuple

import torch


class CurriculumScheduler:
    """
    Three-stage curriculum scheduler.

    Determines which importance range to focus on at each training step
    and computes per-token curriculum weights.

    Args:
        total_steps:   Total training steps (1_000_000).
        warmup_steps:  Linear LR warmup steps (1000).
    """

    # Stage definitions matching the paper exactly
    STAGES = {
        "easy": {
            "importance_range": (0.0, 0.3),
            "fraction": 0.20,  # first 20% of training
            "description": "Focus on low-importance, high-frequency tokens",
        },
        "medium": {
            "importance_range": (0.3, 0.7),
            "fraction": 0.40,  # next 40%
            "description": "Balanced mix of easy and hard tokens",
        },
        "hard": {
            "importance_range": (0.0, 1.0),  # full range
            "fraction": 0.40,  # final 40%
            "description": "Full range – focus on all tokens including hard ones",
        },
    }

    def __init__(
        self,
        total_steps: int = 1_000_000,
        warmup_steps: int = 1000,
        plateau_threshold: float = 0.01,
        plateau_window: int = 10_000,
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.plateau_threshold = plateau_threshold
        self.plateau_window = plateau_window

        # Precompute stage boundaries
        fracs = [self.STAGES[s]["fraction"] for s in ("easy", "medium", "hard")]
        cum = 0.0
        self._boundaries = []
        for f in fracs:
            cum += f
            self._boundaries.append(int(cum * total_steps))
        # _boundaries = [200K, 600K, 1M]

        # Dynamic adaptation bookkeeping
        self._loss_history: list = []
        self._stage_extension: int = 0

    def get_current_stage(self, step: int) -> str:
        """Return the current stage name."""
        effective_step = step - self._stage_extension
        effective_step = max(0, min(effective_step, self.total_steps - 1))

        if effective_step < self._boundaries[0]:
            return "easy"
        elif effective_step < self._boundaries[1]:
            return "medium"
        else:
            return "hard"

    def get_stage_progress(self, step: int) -> float:
        """Smooth cosine interpolation within the current stage (0→1)."""
        stage = self.get_current_stage(step)
        effective = step - self._stage_extension

        if stage == "easy":
            lo, hi = 0, self._boundaries[0]
        elif stage == "medium":
            lo, hi = self._boundaries[0], self._boundaries[1]
        else:
            lo, hi = self._boundaries[1], self._boundaries[2]

        raw = (effective - lo) / max(hi - lo, 1)
        raw = max(0.0, min(1.0, raw))
        # Smooth cosine interpolation
        return 0.5 * (1.0 - math.cos(math.pi * raw))

    def compute_curriculum_weights(
        self,
        importance: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """
        Compute per-token curriculum weights.

        Tokens inside the stage's importance range receive weight 2.0;
        tokens outside receive weight 1.0.  The hard stage uses the
        full range [0, 1] so all tokens get weight 2.0.

        Args:
            importance: (B, L) importance scores.
            step:       Current training step.

        Returns:
            weights: (B, L) curriculum weights.
        """
        stage = self.get_current_stage(step)
        lo, hi = self.STAGES[stage]["importance_range"]
        in_range = (importance >= lo) & (importance <= hi)
        weights = torch.where(in_range, 2.0, 1.0)
        return weights

    def update_loss(self, step: int, loss: float):
        """Record loss for dynamic stage extension."""
        self._loss_history.append((step, loss))
        # Keep only recent history
        if len(self._loss_history) > 50_000:
            self._loss_history = self._loss_history[-50_000:]

    def check_plateau(self, step: int) -> bool:
        """
        Check if loss has plateaued (< 1% decrease over plateau_window steps).
        If so, extend the current stage.
        """
        if len(self._loss_history) < 2 * self.plateau_window:
            return False

        recent = [l for s, l in self._loss_history if s > step - self.plateau_window]
        earlier = [
            l
            for s, l in self._loss_history
            if step - 2 * self.plateau_window < s <= step - self.plateau_window
        ]

        if not recent or not earlier:
            return False

        avg_recent = sum(recent) / len(recent)
        avg_earlier = sum(earlier) / len(earlier)
        relative_change = (avg_earlier - avg_recent) / (abs(avg_earlier) + 1e-8)

        if relative_change < self.plateau_threshold:
            self._stage_extension += self.plateau_window
            return True
        return False

    def get_state(self, step: int) -> Dict:
        """Return current curriculum state for logging."""
        stage = self.get_current_stage(step)
        return {
            "curriculum/stage": stage,
            "curriculum/progress": self.get_stage_progress(step),
            "curriculum/importance_lo": self.STAGES[stage]["importance_range"][0],
            "curriculum/importance_hi": self.STAGES[stage]["importance_range"][1],
        }
