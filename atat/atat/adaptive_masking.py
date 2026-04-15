"""
Adaptive Masking Scheduler (§3.2)

Implements the balanced masking curriculum from the paper:

    g_bal(i, t) = t · g_inv(i) + (1-t) · g_prop(i)

where:
    g_prop(i) = η + (1-2η) · i          (importance-proportional)
    g_inv(i)  = η + (1-2η) · (1-i)      (importance-inverse)

with smoothing η = 0.3, yielding masking probabilities in [0.3, 0.7].

Position-dependent masking probability:
    q_ATAT(z^l_t = [M] | x^l, i^l) = (1 - α_t) · g_bal(i^l, t)

Curriculum interpretation:
    t → 1 (heavy corruption): g_bal ≈ g_inv → preserve important tokens as anchors
    t → 0 (light corruption): g_bal ≈ g_prop → focus training on important tokens
"""

import torch
import torch.nn as nn
from typing import Optional


class AdaptiveMaskingScheduler(nn.Module):
    """
    Balanced masking scheduler with time-varying importance weighting.

    Strategies (for ablations in Table 3):
        "balanced"     – time-varying interpolation (default, Eq. 5-6)
        "proportional" – importance-proportional only
        "inverse"      – importance-inverse only
        "uniform"      – standard uniform masking (baseline)

    Args:
        strategy:  Masking strategy name.
        eta:       Smoothing parameter η (0.3).
    """

    VALID_STRATEGIES = ("balanced", "proportional", "inverse", "uniform")

    def __init__(
        self,
        strategy: str = "balanced",
        eta: float = 0.3,
    ):
        super().__init__()
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy}'. Must be one of {self.VALID_STRATEGIES}"
            )
        self.strategy = strategy
        self.eta = eta

    def g_proportional(self, importance: torch.Tensor) -> torch.Tensor:
        """g_prop(i) = η + (1-2η)·i"""
        return self.eta + (1.0 - 2.0 * self.eta) * importance

    def g_inverse(self, importance: torch.Tensor) -> torch.Tensor:
        """g_inv(i) = η + (1-2η)·(1-i)"""
        return self.eta + (1.0 - 2.0 * self.eta) * (1.0 - importance)

    def g_balanced(
        self, importance: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """g_bal(i, t) = t·g_inv(i) + (1-t)·g_prop(i)"""
        t = self._broadcast_t(t, importance)
        return t * self.g_inverse(importance) + (1.0 - t) * self.g_proportional(importance)

    def compute_masking_probabilities(
        self,
        importance: torch.Tensor,
        t: torch.Tensor,
        alpha_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token masking probabilities.

            P(mask | x^l, i^l) = (1 - α_t) · g(i^l, t)

        Args:
            importance: (B, L) importance scores in [0, 1].
            t:          (B,) or scalar diffusion timestep in [0, 1].
            alpha_t:    (B,) or scalar noise schedule value.

        Returns:
            mask_probs: (B, L) masking probabilities.
        """
        if self.strategy == "uniform":
            alpha_t_broad = self._broadcast_t(alpha_t, importance)
            return (1.0 - alpha_t_broad).expand_as(importance)

        if self.strategy == "proportional":
            g = self.g_proportional(importance)
        elif self.strategy == "inverse":
            g = self.g_inverse(importance)
        else:  # balanced
            g = self.g_balanced(importance, t)

        alpha_t_broad = self._broadcast_t(alpha_t, importance)
        mask_probs = (1.0 - alpha_t_broad) * g
        return mask_probs.clamp(0.0, 1.0)

    def sample_masks(
        self,
        input_ids: torch.Tensor,
        importance: torch.Tensor,
        t: torch.Tensor,
        alpha_t: torch.Tensor,
        mask_token_id: int,
    ) -> torch.Tensor:
        """
        Sample masked input for the forward diffusion process.

        Args:
            input_ids:     (B, L) clean token IDs.
            importance:    (B, L) importance scores.
            t:             Diffusion timestep.
            alpha_t:       Noise schedule α_t.
            mask_token_id: Token ID for [MASK].

        Returns:
            masked_ids: (B, L) with some tokens replaced by mask_token_id.
        """
        mask_probs = self.compute_masking_probabilities(importance, t, alpha_t)
        mask_decisions = torch.bernoulli(mask_probs).bool()
        masked_ids = input_ids.clone()
        masked_ids[mask_decisions] = mask_token_id
        return masked_ids

    @staticmethod
    def _broadcast_t(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Broadcast scalar/1D timestep to (B, 1) for element-wise ops."""
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        return t
