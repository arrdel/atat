"""
ATAT Model Architecture
Full model: frozen GPT-2 anchor → importance estimator → denoiser.

"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model

from atat.atat.importance_estimator import ImportanceEstimator
from atat.atat.adaptive_masking import AdaptiveMaskingScheduler
from atat.atat.curriculum import CurriculumScheduler
from atat.atat.uncertainty_sampler import UncertaintySampler


# ---------------------------------------------------------------------------
# Building blocks (adapted from MDLM DiT, self-contained)
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """Pre-LayerNorm wrapper."""
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class Rotary(nn.Module):
    """RoPE positional encoding."""
    def __init__(self, dim: int, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor, seq_dim: int = 1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
        return self.cos_cached, self.sin_cached


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE to query and key tensors."""
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embeddings → MLP."""
    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_dim = freq_dim

    @staticmethod
    def sinusoidal_embedding(t, dim, max_period=10_000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t):
        return self.mlp(self.sinusoidal_embedding(t, self.freq_dim))


# ---------------------------------------------------------------------------
# Transformer block with adaptive layer norm (DiT-style)
# ---------------------------------------------------------------------------

class DenoiserBlock(nn.Module):
    """
    Single Transformer block for the ATAT denoiser.

    Pre-LayerNorm, multi-head self-attention with RoPE,
    GELU feed-forward, adaLN conditioning on timestep.

    Args:
        hidden_size: Model dimension (768).
        n_heads:     Number of attention heads (12).
        cond_dim:    Conditioning dimension (768).
        mlp_ratio:   FFN expansion ratio (4 → 3072).
        dropout:     Dropout rate (0.1).
    """

    def __init__(
        self,
        hidden_size: int = 768,
        n_heads: int = 12,
        cond_dim: int = 768,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        # Pre-norm + attention
        self.norm1 = LayerNorm(hidden_size)
        self.attn_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        # Pre-norm + FFN
        self.norm2 = LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_ratio * hidden_size),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_size, hidden_size),
        )
        self.dropout2 = nn.Dropout(dropout)

        # adaLN: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * hidden_size)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, L, D = x.shape

        # adaLN modulation
        mod = self.adaLN_modulation(c).unsqueeze(1)  # (B, 1, 6*D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
            6, dim=-1
        )

        # Self-attention with pre-norm
        x_norm = self.norm1(x) * (1 + scale_msa) + shift_msa
        qkv = self.attn_qkv(x_norm).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, L, H, D_h)

        # Apply RoPE to q and k
        q = q.transpose(1, 2)  # (B, H, L, D_h)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Expand cos/sin for heads: (L, D_h) → (1, 1, L, D_h)
        cos_exp = cos[:L].unsqueeze(0).unsqueeze(0)
        sin_exp = sin[:L].unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos_exp, sin_exp)

        # Scaled dot-product attention (uses Flash Attention if available)
        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout1.p if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).reshape(B, L, D)

        x = x + gate_msa * self.dropout1(self.attn_out(attn))

        # FFN with pre-norm
        x_norm2 = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.dropout2(self.mlp(x_norm2))

        return x


class FinalLayer(nn.Module):
    """Final normalization + linear projection to vocabulary."""

    def __init__(self, hidden_size: int, vocab_size: int, cond_dim: int):
        super().__init__()
        self.norm = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).unsqueeze(1).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return self.linear(x)


# ---------------------------------------------------------------------------
# Full ATAT Model
# ---------------------------------------------------------------------------


class ATATDiT(nn.Module):
    """
    Full ATAT model: frozen GPT-2 anchor + importance estimator + denoiser.

    Paper §3 architecture:
      1. Frozen GPT-2 Small (124M) provides hidden states h^l for each token.
      2. ImportanceEstimator (200K) maps h^l → importance score i^l.
      3. Importance projection W_i ∈ R^{768×2}: input [i^l; 1-i^l].
      4. Denoiser: 6-layer Transformer (48M) with RoPE, pre-LN, adaLN,
         initialized from GPT-2 layers 0-5.

    Args:
        vocab_size:    Vocabulary size (50257).
        hidden_size:   Hidden dimension (768).
        n_heads:       Number of attention heads (12).
        n_layers:      Number of denoiser layers (6).
        cond_dim:      Conditioning dimension (768).
        mlp_ratio:     FFN expansion ratio (4).
        dropout:       Dropout rate (0.1).
        mask_token_id: Mask token id (50256 = GPT-2 EOS repurposed).
        importance_mode: ImportanceEstimator mode.
        lambda_hybrid: Hybrid weight (0.7).
        gamma:         Importance loss weight (0.003).
        beta:          Importance weighting scale (1.0).
        eta:           Masking smoothing (0.3).
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        n_heads: int = 12,
        n_layers: int = 6,
        cond_dim: int = 768,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        mask_token_id: int = 50256,
        importance_mode: str = "full",
        lambda_hybrid: float = 0.7,
        gamma: float = 0.003,
        beta: float = 1.0,
        eta: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.mask_token_id = mask_token_id
        self.gamma = gamma
        self.beta = beta

        # ---- 1. Frozen GPT-2 anchor (124M) ----
        self.anchor = GPT2Model.from_pretrained("gpt2")
        for param in self.anchor.parameters():
            param.requires_grad = False
        self.anchor.eval()

        # ---- 2. Importance estimator (200K) ----
        self.importance_estimator = ImportanceEstimator(
            hidden_dim=hidden_size,
            mlp_hidden=256,
            mode=importance_mode,
            lambda_hybrid=lambda_hybrid,
        )

        # ---- 3. Importance projection W_i ∈ R^{768×2} ----
        # Input: [i^l; 1-i^l] → 768-dim
        self.importance_projection = nn.Linear(2, hidden_size)

        # ---- 4. Token embedding (trainable) ----
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        nn.init.normal_(self.token_embedding.weight, std=0.02)

        # ---- 5. Timestep embedding ----
        self.sigma_map = TimestepEmbedder(cond_dim)

        # ---- 6. RoPE ----
        self.rotary_emb = Rotary(hidden_size // n_heads)

        # ---- 7. Denoiser: 6-layer Transformer (48M) ----
        self.blocks = nn.ModuleList([
            DenoiserBlock(
                hidden_size=hidden_size,
                n_heads=n_heads,
                cond_dim=cond_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # ---- 8. Output layer ----
        self.output_layer = FinalLayer(hidden_size, vocab_size, cond_dim)

        # ---- Masking scheduler ----
        self.masking_scheduler = AdaptiveMaskingScheduler(
            strategy="balanced",
            eta=eta,
        )

        # ---- Initialize denoiser from GPT-2 layers 0-5 ----
        self._init_from_gpt2()

    def _init_from_gpt2(self):
        """Initialize the 6 denoiser blocks from GPT-2 layers 0-5."""
        gpt2_blocks = self.anchor.h  # nn.ModuleList of GPT-2 transformer blocks

        for i, block in enumerate(self.blocks):
            if i >= len(gpt2_blocks):
                break
            gpt2_block = gpt2_blocks[i]

            with torch.no_grad():
                # GPT-2 uses Conv1D where weight shape is (in, out)
                # Q, K, V projection
                gpt2_qkv_weight = gpt2_block.attn.c_attn.weight.data  # (768, 2304)
                block.attn_qkv.weight.copy_(gpt2_qkv_weight.T)

                # Output projection
                gpt2_out_weight = gpt2_block.attn.c_proj.weight.data  # (768, 768)
                block.attn_out.weight.copy_(gpt2_out_weight.T)

                # FFN layer 1
                gpt2_ffn1_weight = gpt2_block.mlp.c_fc.weight.data  # (768, 3072)
                gpt2_ffn1_bias = gpt2_block.mlp.c_fc.bias.data
                block.mlp[0].weight.copy_(gpt2_ffn1_weight.T)
                block.mlp[0].bias.copy_(gpt2_ffn1_bias)

                # FFN layer 2
                gpt2_ffn2_weight = gpt2_block.mlp.c_proj.weight.data  # (3072, 768)
                gpt2_ffn2_bias = gpt2_block.mlp.c_proj.bias.data
                block.mlp[2].weight.copy_(gpt2_ffn2_weight.T)
                block.mlp[2].bias.copy_(gpt2_ffn2_bias)

                # LayerNorms
                block.norm1.weight.copy_(gpt2_block.ln_1.weight.data)
                block.norm2.weight.copy_(gpt2_block.ln_2.weight.data)

        # Copy GPT-2 token embedding to our token embedding
        with torch.no_grad():
            self.token_embedding.weight.copy_(self.anchor.wte.weight.data)

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_importance(
        self,
        input_ids: torch.Tensor,
        frequency_table: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute per-token importance scores using the frozen anchor.

        Args:
            input_ids: (B, L) token ids.
            frequency_table: Optional (V,) frequency table.

        Returns:
            importance: (B, L) in [0, 1].
        """
        with torch.no_grad():
            self.anchor.eval()
            hidden_states = self.anchor(input_ids).last_hidden_state  # (B, L, 768)

        return self.importance_estimator(
            hidden_states,
            input_ids=input_ids,
            frequency_table=frequency_table,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        sigma: torch.Tensor,
        importance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Denoiser forward pass.

        Args:
            x_t:        (B, L) noisy token ids.
            sigma:      (B,) diffusion timestep.
            importance: (B, L) precomputed importance scores.

        Returns:
            logits: (B, L, V) predicted token logits.
        """
        B, L = x_t.shape

        # Token embedding
        x = self.token_embedding(x_t)  # (B, L, D)

        # Add importance information if available
        if importance is not None:
            imp_input = torch.stack(
                [importance, 1.0 - importance], dim=-1
            )  # (B, L, 2)
            x = x + self.importance_projection(imp_input)  # (B, L, D)

        # Timestep conditioning
        c = F.silu(self.sigma_map(sigma))  # (B, D)

        # RoPE
        cos, sin = self.rotary_emb(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, cos, sin)

        # Output projection
        logits = self.output_layer(x, c)  # (B, L, V)
        return logits

    def compute_loss(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        importance: torch.Tensor,
        oracle_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the ATAT training loss (Eq. 5):

            L_ATAT = L_denoise + γ · L_importance

        where L_denoise uses importance-weighted cross-entropy:
            w(i) = 1 + β · i

        Args:
            x_0:            (B, L) clean tokens.
            x_t:            (B, L) noisy tokens.
            t:              (B,) timestep.
            importance:     (B, L) importance scores.
            oracle_targets: (B, L) oracle importance targets (for MSE loss).

        Returns:
            Dictionary with 'total', 'denoise', 'importance' losses.
        """
        # Forward pass through denoiser
        logits = self.forward(x_t, t, importance)  # (B, L, V)

        # Importance-weighted cross-entropy: w(i) = 1 + β · i
        per_token_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            x_0.reshape(-1),
            reduction="none",
        ).reshape(x_0.shape)  # (B, L)

        weights = 1.0 + self.beta * importance  # (B, L)

        # Only compute loss on masked positions
        is_masked = (x_t == self.mask_token_id).float()
        denoise_loss = (per_token_loss * weights * is_masked).sum() / (
            is_masked.sum() + 1e-8
        )

        # Importance estimation loss: MSE against oracle
        if oracle_targets is not None:
            importance_loss = self.importance_estimator.compute_loss(
                importance, oracle_targets
            )
        else:
            importance_loss = torch.tensor(0.0, device=x_0.device)

        total_loss = denoise_loss + self.gamma * importance_loss

        return {
            "total": total_loss,
            "denoise": denoise_loss,
            "importance": importance_loss,
        }

    def adaptive_mask(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        importance: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply adaptive masking using the balanced schedule g_bal.

        Args:
            x_0:        (B, L) clean tokens.
            t:          (B,) diffusion timestep in [0, 1].
            importance: (B, L) importance scores.

        Returns:
            x_t: (B, L) masked tokens.
        """
        mask_probs = self.masking_scheduler.compute_mask_probabilities(
            importance, t
        )  # (B, L)
        mask_decisions = torch.rand_like(mask_probs) < mask_probs
        x_t = torch.where(mask_decisions, self.mask_token_id, x_0)
        return x_t
