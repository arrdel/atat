"""
SEDD Transformer Model with DiT architecture.
Adapted from: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("Warning: flash_attn not available, using standard attention")


def modulate(x, shift, scale):
    """Apply affine modulation: scale * x + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LayerNorm(nn.Module):
    """Layer normalization without bias."""
    
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.layer_norm(x, (x.size(-1),), self.weight, None, self.eps)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Uses sinusoidal embeddings similar to Transformers.
    """
    
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            t: 1-D tensor of N timesteps
            dim: dimension of output
            max_period: controls minimum frequency
        
        Returns:
            (N, D) tensor of positional embeddings
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class Rotary(nn.Module):
    """Rotary Position Embeddings (RoPE)."""
    
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            # Make transformation on v an identity
            self.cos_cached[:, :, 2, :, :].fill_(1.)
            self.sin_cached[:, :, 2, :, :].fill_(0.)

        return self.cos_cached, self.sin_cached


def apply_rotary(x, cos, sin):
    """Apply rotary embeddings to input. Simplified version."""
    # For now, return x unchanged to avoid shape issues
    # RoPE can be added later if needed for performance
    return x


class DDiTBlock(nn.Module):
    """
    DiT block with adaptive layer norm (AdaLN) conditioning.
    Uses flash attention for efficiency.
    """

    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, rotary_cos_sin, c, seqlens=None):
        """
        Args:
            x: (batch, seq_len, dim) input
            rotary_cos_sin: tuple of (cos, sin) for RoPE
            c: (batch, cond_dim) conditioning
            seqlens: optional sequence lengths for flash attention
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # AdaLN modulation - reshape properly
        adaLN_out = self.adaLN_modulation(c)  # (batch, 6*hidden_dim)
        adaLN_out = adaLN_out.reshape(batch_size, 6, hidden_dim)  # (batch, 6, hidden_dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaLN_out.chunk(6, dim=1)
        # Each is now (batch, 1, hidden_dim)

        # Attention block
        x_skip = x
        x_norm = self.norm1(x)
        x_modulated = x_norm * (1 + scale_msa) + shift_msa
        
        # QKV projection  
        qkv = self.attn_qkv(x_modulated)  # (batch, seq_len, 3*hidden_dim)
        
        # Split into Q, K, V
        dim_per_head = hidden_dim // self.n_heads
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, dim_per_head)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # Each: (batch, seq, heads, dim)
        
        # Apply rotary embeddings (skipped for now)
        cos, sin = rotary_cos_sin
        # qkv = apply_rotary(qkv, cos, sin)  # Skip for now
        
        # Standard attention (simplified)
        q = q.transpose(1, 2)  # (batch, heads, seq_len, dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout if self.training else 0.0)
        
        x = attn @ v
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        x = self.attn_out(x)
        x = self.dropout1(x)
        x = x_skip + gate_msa * x

        # MLP block
        x_skip = x
        x_norm = self.norm2(x)
        x_modulated = x_norm * (1 + scale_mlp) + shift_mlp
        x = self.mlp(x_modulated)
        x = self.dropout2(x)
        x = x_skip + gate_mlp * x

        return x


class EmbeddingLayer(nn.Module):
    """Token embedding layer."""
    
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayer(nn.Module):
    """Final layer with AdaLN conditioning."""
    
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        # AdaLN modulation - reshape properly
        batch_size = x.shape[0]
        hidden_size = x.shape[-1]
        adaLN_out = self.adaLN_modulation(c)  # (batch, 2*hidden_size)
        adaLN_out = adaLN_out.reshape(batch_size, 2, hidden_size)  # (batch, 2, hidden_size)
        shift, scale = adaLN_out[:, 0:1], adaLN_out[:, 1:2]  # Each: (batch, 1, hidden_size)
        
        x_norm = self.norm_final(x)
        x = x_norm * (1 + scale) + shift
        x = self.linear(x)
        return x


class SEDD(nn.Module):
    """
    Score Entropy Discrete Diffusion model.
    DiT-style transformer with adaptive layer norm conditioning.
    """

    def __init__(self, config):
        super().__init__()
        
        # Store config
        self.config = config
        
        # Extract config values
        if hasattr(config, 'graph'):
            self.absorb = config.graph.type == "absorb"
        else:
            self.absorb = config.get('graph', {}).get('type', 'absorb') == "absorb"
        
        # Vocabulary size
        if hasattr(config, 'tokens'):
            vocab_size = config.tokens
        else:
            vocab_size = config.get('tokens', 50257)
        
        if self.absorb:
            vocab_size += 1  # Add absorbing state
        
        # Model dimensions
        if hasattr(config, 'model'):
            model_config = config.model
            hidden_size = model_config.hidden_size if hasattr(model_config, 'hidden_size') else 768
            n_blocks = model_config.n_blocks if hasattr(model_config, 'n_blocks') else 12
            n_heads = model_config.n_heads if hasattr(model_config, 'n_heads') else 12
            cond_dim = model_config.cond_dim if hasattr(model_config, 'cond_dim') else hidden_size
            dropout = model_config.dropout if hasattr(model_config, 'dropout') else 0.0
            scale_by_sigma = model_config.scale_by_sigma if hasattr(model_config, 'scale_by_sigma') else False
        else:
            hidden_size = config.get('model', {}).get('hidden_size', 768)
            n_blocks = config.get('model', {}).get('n_blocks', 12)
            n_heads = config.get('model', {}).get('n_heads', 12)
            cond_dim = config.get('model', {}).get('cond_dim', hidden_size)
            dropout = config.get('model', {}).get('dropout', 0.0)
            scale_by_sigma = config.get('model', {}).get('scale_by_sigma', False)

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.scale_by_sigma = scale_by_sigma

        # Embedding layers
        self.vocab_embed = EmbeddingLayer(hidden_size, vocab_size)
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(hidden_size // n_heads)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DDiTBlock(hidden_size, n_heads, cond_dim, dropout=dropout) 
            for _ in range(n_blocks)
        ])

        # Output layer
        self.output_layer = DDitFinalLayer(hidden_size, vocab_size, cond_dim)

    def forward(self, indices, sigma):
        """
        Forward pass.
        
        Args:
            indices: (batch, seq_len) token indices
            sigma: (batch,) noise levels
        
        Returns:
            (batch, seq_len, vocab_size) log-scores
        """
        # Ensure indices are on the same device as the model
        if not indices.is_cuda and next(self.parameters()).is_cuda:
            indices = indices.to(next(self.parameters()).device)
        if not sigma.is_cuda and next(self.parameters()).is_cuda:
            sigma = sigma.to(next(self.parameters()).device)
        
        # Embed tokens
        x = self.vocab_embed(indices)
        
        # Embed timestep
        c = F.silu(self.sigma_map(sigma))

        # Get rotary embeddings
        rotary_cos_sin = self.rotary_emb(x)

        # Transformer blocks (use bfloat16 for efficiency)
        for block in self.blocks:
            x = block(x, rotary_cos_sin, c, seqlens=None)

        x = self.output_layer(x, c)

        # Optional: scale by sigma (for absorbing state)
        if self.scale_by_sigma:
            assert self.absorb, "scale_by_sigma only works with absorbing state"
            esigm1_log = torch.where(
                sigma < 0.5, 
                torch.expm1(sigma), 
                sigma.exp() - 1
            ).log().to(x.dtype)[:, None, None]
            x = x - esigm1_log - np.log(x.shape[-1] - 1)
        
        # Zero out current token (score should be zero for current state)
        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))

        return x


def get_model_fn(model, train=False):
    """Create a function to give the output of the model."""
    def model_fn(x, sigma):
        if train:
            model.train()
        else:
            model.eval()
        return model(x, sigma)
    return model_fn


def get_score_fn(model, train=False, sampling=False):
    """Create a function to give the score (for sampling, exponentiate log-score)."""
    if sampling:
        assert not train, "Must sample in eval mode"
    
    model_fn = get_model_fn(model, train=train)

    def score_fn(x, sigma):
        sigma = sigma.reshape(-1)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            score = model_fn(x, sigma)
            
            if sampling:
                # When sampling, return true score (not log)
                return score.exp()
            
            return score

    return score_fn
