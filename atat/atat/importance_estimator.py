"""
Importance Estimator Module
Implements the hybrid importance estimator that combines learned contextual signals with a frequency-based prior.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImportanceEstimator(nn.Module):
    """
    Token-level importance estimator.

    Operates on frozen GPT-2 hidden states and predicts per-token importance
    scores in [0, 1].  Four modes:

        "full"           – λ·learned + (1-λ)·frequency  (paper default)
        "learned_only"   – learned component only (λ=1)
        "frequency_only" – frequency prior only (λ=0)
        "uniform"        – constant 0.5 (no importance signal)

    Architecture (Table 8):
        LayerNorm(768) → Linear(768, 256) → GELU → Linear(256, 1) → Sigmoid
        ≈ 200K parameters  (768×256 + 256 + 256×1 + 1 + LN params = 198,657)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        input_dim: int = 768,
        lambda_blend: float = 0.7,
        mode: str = "full",
        vocab_size: int = 50257,
        oracle_tau: float = 10.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.lambda_blend = lambda_blend
        self.mode = mode
        self.vocab_size = vocab_size
        self.oracle_tau = oracle_tau

        # Learned contextual estimator (2-layer MLP)
        if mode in ("full", "learned_only"):
            self.mlp = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        # Frequency prior table (registered buffer, not a parameter)
        self.register_buffer(
            "freq_table",
            torch.zeros(vocab_size),
            persistent=False,
        )
        self._freq_loaded = False

    # ------------------------------------------------------------------
    # Frequency table management
    # ------------------------------------------------------------------
    def load_frequency_table(self, path_or_tensor):
        """Load a precomputed token frequency table.

        Args:
            path_or_tensor: Either a file path (str/Path) or a 1-D tensor
                            of shape (vocab_size,) with raw counts.
        """
        if isinstance(path_or_tensor, (str,)):
            freq = torch.load(path_or_tensor, map_location="cpu")
        else:
            freq = path_or_tensor

        if freq.shape[0] != self.vocab_size:
            raise ValueError(
                f"Frequency table size {freq.shape[0]} != vocab_size {self.vocab_size}"
            )

        # Convert raw counts → importance: i_freq = 1 - log(f+1)/log(max_f+1)
        log_freq = torch.log(freq.float() + 1.0)
        log_max = log_freq.max()
        importance = 1.0 - log_freq / (log_max + 1e-8)
        self.freq_table.copy_(importance)
        self._freq_loaded = True

    def frequency_importance(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up frequency-based importance for token IDs.

        Args:
            token_ids: (batch, seq_len) int tensor.
        Returns:
            (batch, seq_len) float tensor in [0, 1].
        """
        return self.freq_table[token_ids]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-token importance scores.

        Args:
            hidden_states: Frozen GPT-2 hidden states, shape (B, L, 768).
            token_ids:     Original token IDs, shape (B, L).  Required when
                           mode uses frequency.

        Returns:
            Importance scores, shape (B, L), values in [0, 1].
        """
        B, L, _ = hidden_states.shape

        if self.mode == "uniform":
            return torch.full((B, L), 0.5, device=hidden_states.device)

        if self.mode == "frequency_only":
            if token_ids is None:
                raise ValueError("token_ids required for frequency_only mode")
            return self.frequency_importance(token_ids)

        # Learned component
        i_learned = self.mlp(hidden_states).squeeze(-1)  # (B, L)

        if self.mode == "learned_only":
            return i_learned

        # Full hybrid mode
        if token_ids is None:
            raise ValueError("token_ids required for full mode")
        i_freq = self.frequency_importance(token_ids)  # (B, L)

        return self.lambda_blend * i_learned + (1.0 - self.lambda_blend) * i_freq

    # ------------------------------------------------------------------
    # Oracle targets (§3.2, Eq. 4)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_oracle_targets(
        self,
        anchor_model: nn.Module,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute oracle importance targets from GPT-2 surprisal.

        i^{l,*} = min(1, -log p_GPT2(x^l | x^{<l}) / τ)

        Args:
            anchor_model: Frozen GPT-2 model with a .forward() that returns
                          logits of shape (B, L, V).
            input_ids:    Token IDs, shape (B, L).

        Returns:
            Oracle targets, shape (B, L), values in [0, 1].
        """
        logits = anchor_model(input_ids).logits  # (B, L, V)
        # Shift: logits[:, :-1] predict input_ids[:, 1:]
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        target_ids = input_ids[:, 1:].unsqueeze(-1)
        token_log_probs = log_probs.gather(-1, target_ids).squeeze(-1)

        surprisal = -token_log_probs  # positive
        oracle = torch.clamp(surprisal / self.oracle_tau, max=1.0)

        # Pad first position with median importance
        pad = oracle.median(dim=-1, keepdim=True).values
        oracle = torch.cat([pad, oracle], dim=1)
        return oracle

    # ------------------------------------------------------------------
    # Importance MSE loss
    # ------------------------------------------------------------------
    def importance_loss(
        self,
        predicted: torch.Tensor,
        oracle: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """MSE loss between predicted and oracle importance.

        Args:
            predicted: Predicted importance, (B, L).
            oracle:    Oracle targets, (B, L).
            mask:      Optional boolean mask, (B, L). True = include.

        Returns:
            Scalar MSE loss.
        """
        loss = F.mse_loss(predicted, oracle, reduction="none")
        if mask is not None:
            loss = loss * mask.float()
            return loss.sum() / mask.float().sum().clamp(min=1.0)
        return loss.mean()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_stats(self, importance_scores: torch.Tensor) -> Dict[str, float]:
        """Return summary statistics for logging."""
        return {
            "mean": importance_scores.mean().item(),
            "std": importance_scores.std().item(),
            "min": importance_scores.min().item(),
            "max": importance_scores.max().item(),
            "median": importance_scores.median().item(),
        }
