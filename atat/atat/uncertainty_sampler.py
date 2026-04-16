"""
Uncertainty-Guided Decoding 
Implements the token-level uncertainty sampling strategy used during
ATAT inference.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class UncertaintySampler:
    """
    Uncertainty-guided decoding for ATAT.

    During reverse diffusion, at each step t the sampler:
      1. Computes entropy of the denoiser logits at every masked position.
      2. Multiplies entropy by importance to get priority scores.
      3. Selects the top-k positions to unmask.
      4. Applies temperature sharpening and greedy decoding.

    Args:
        total_steps:     Number of discrete reverse diffusion steps T (NFE).
        vocab_size:      Vocabulary size (50257 for GPT-2).
        mask_token_id:   ID of the [MASK] token.
    """

    def __init__(
        self,
        total_steps: int = 1000,
        vocab_size: int = 50257,
        mask_token_id: int = 50256,
    ):
        self.total_steps = total_steps
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute token-level entropy H(p_θ).

        Args:
            logits: (B, L, V) denoiser output logits.

        Returns:
            entropy: (B, L)
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, L)
        return entropy

    def compute_priority(
        self,
        logits: torch.Tensor,
        importance: torch.Tensor,
    ) -> torch.Tensor:
        """
        Priority score: s^l = u^l · i^l.

        Args:
            logits:     (B, L, V) denoiser logits.
            importance: (B, L)    importance scores.

        Returns:
            priority: (B, L) – higher means "unmask this first".
        """
        entropy = self.compute_entropy(logits)
        return entropy * importance

    def compute_sharpening_temperature(self, t: int) -> float:
        """
        κ = 1 / (t/T + 0.1)

        Args:
            t: Current reverse diffusion step (t=T is most noisy).

        Returns:
            κ (sharpening temperature).
        """
        return 1.0 / (t / self.total_steps + 0.1)

    def compute_num_unmask(
        self,
        alpha_t: float,
        alpha_t_next: float,
        seq_len: int,
    ) -> int:
        """
        k = ⌊ L · |α_{t+Δt} − α_t| ⌋

        Args:
            alpha_t:      Noise rate at current step.
            alpha_t_next: Noise rate at next step.
            seq_len:      Sequence length L.

        Returns:
            Number of tokens to unmask this step.
        """
        k = int(seq_len * abs(alpha_t_next - alpha_t))
        return max(k, 1)  # unmask at least 1 token

    @torch.no_grad()
    def sample_step(
        self,
        x_t: torch.Tensor,
        logits: torch.Tensor,
        importance: torch.Tensor,
        t: int,
        alpha_t: float,
        alpha_t_next: float,
    ) -> torch.Tensor:
        """
        One step of uncertainty-guided decoding.

        Args:
            x_t:        (B, L) current noisy sequence (contains mask_token_id).
            logits:     (B, L, V) denoiser output logits.
            importance: (B, L) importance scores.
            t:          Current reverse diffusion step.
            alpha_t:    Current noise rate.
            alpha_t_next: Next noise rate.

        Returns:
            x_next: (B, L) sequence with top-k positions unmasked.
        """
        B, L = x_t.shape

        # 1. Priority scores for masked positions only
        is_masked = (x_t == self.mask_token_id)  # (B, L)
        priority = self.compute_priority(logits, importance)  # (B, L)
        # Zero out priority for already-unmasked positions
        priority = priority * is_masked.float()

        # 2. Number of tokens to unmask
        k = self.compute_num_unmask(alpha_t, alpha_t_next, L)

        # 3. Select top-k positions per batch element
        _, top_indices = priority.topk(min(k, is_masked.sum(dim=-1).min().item()), dim=-1)

        # 4. Sharpen logits and greedy decode
        kappa = self.compute_sharpening_temperature(t)
        sharpened_logits = logits * kappa  # (B, L, V)
        predictions = sharpened_logits.argmax(dim=-1)  # (B, L)

        # 5. Unmask the top-k positions
        x_next = x_t.clone()
        # Scatter predictions into selected positions
        batch_idx = torch.arange(B, device=x_t.device).unsqueeze(1).expand_as(top_indices)
        x_next[batch_idx, top_indices] = predictions[batch_idx, top_indices]

        return x_next

    @torch.no_grad()
    def generate(
        self,
        model,
        importance_estimator,
        seq_len: int,
        batch_size: int = 1,
        device: str = "cuda",
        noise_schedule_fn=None,
    ) -> torch.Tensor:
        """
        Full reverse diffusion generation loop.

        Args:
            model:                 ATAT denoiser (callable: x_t → logits).
            importance_estimator:  Importance estimator module.
            seq_len:               Target sequence length.
            batch_size:            Number of sequences to generate.
            device:                Device.
            noise_schedule_fn:     Function step → (alpha_t, alpha_t_next).

        Returns:
            x_0: (B, L) generated token ids.
        """
        # Start fully masked
        x_t = torch.full(
            (batch_size, seq_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )

        for t in range(self.total_steps, 0, -1):
            # Get noise rates
            if noise_schedule_fn is not None:
                alpha_t, alpha_t_next = noise_schedule_fn(t)
            else:
                # Default log-linear: alpha_t = 1 - t/T
                alpha_t = 1.0 - t / self.total_steps
                alpha_t_next = 1.0 - (t - 1) / self.total_steps

            # Forward pass
            logits = model(x_t, t=t)
            importance = importance_estimator.estimate(x_t)

            # One sampling step
            x_t = self.sample_step(
                x_t=x_t,
                logits=logits,
                importance=importance,
                t=t,
                alpha_t=alpha_t,
                alpha_t_next=alpha_t_next,
            )

        return x_t
