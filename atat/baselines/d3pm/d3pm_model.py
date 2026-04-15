"""
D3PM (Discrete Denoising Diffusion Probabilistic Models) implementation.

Based on: "Structured Denoising Diffusion Models in Discrete State-Spaces"
Austin et al., NeurIPS 2021

This is a simplified implementation for text generation using absorbing state diffusion.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class D3PM(nn.Module):
    """
    D3PM model for discrete text generation.
    Uses a standard transformer with absorbing state diffusion.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Model dimensions
        if hasattr(config, 'model'):
            model_config = config.model
            self.vocab_size = model_config.vocab_size if hasattr(model_config, 'vocab_size') else 50257
            self.hidden_size = model_config.hidden_size if hasattr(model_config, 'hidden_size') else 768
            self.num_layers = model_config.num_layers if hasattr(model_config, 'num_layers') else 12
            self.num_heads = model_config.num_heads if hasattr(model_config, 'num_heads') else 12
            self.max_length = model_config.max_length if hasattr(model_config, 'max_length') else 1024
            self.dropout = model_config.dropout if hasattr(model_config, 'dropout') else 0.1
        else:
            self.vocab_size = config.get('vocab_size', 50257)
            self.hidden_size = config.get('hidden_size', 768)
            self.num_layers = config.get('num_layers', 12)
            self.num_heads = config.get('num_heads', 12)
            self.max_length = config.get('max_length', 1024)
            self.dropout = config.get('dropout', 0.1)
        
        # Add mask token to vocabulary
        self.vocab_size_with_mask = self.vocab_size + 1
        self.mask_token_id = self.vocab_size  # Last token is mask
        
        # Token embeddings
        self.token_embedding = nn.Embedding(self.vocab_size_with_mask, self.hidden_size)
        
        # Position embeddings
        self.position_embedding = nn.Embedding(self.max_length, self.hidden_size)
        
        # Timestep embedding
        self.time_embed = TimestepEmbedding(self.hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ])
        
        # Layer norm
        self.ln_f = nn.LayerNorm(self.hidden_size)
        
        # Output projection to vocabulary
        self.output_proj = nn.Linear(self.hidden_size, self.vocab_size_with_mask, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x, t):
        """
        Forward pass.
        
        Args:
            x: (batch_size, seq_len) token indices (can include mask tokens)
            t: (batch_size,) diffusion timesteps in [0, num_timesteps)
        
        Returns:
            logits: (batch_size, seq_len, vocab_size_with_mask) predicted logits
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Token embeddings
        token_emb = self.token_embedding(x)  # (batch, seq, hidden)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)  # (batch, seq, hidden)
        
        # Timestep embedding
        time_emb = self.time_embed(t)  # (batch, hidden)
        time_emb = time_emb.unsqueeze(1)  # (batch, 1, hidden)
        
        # Combine embeddings
        h = token_emb + pos_emb + time_emb
        
        # Transformer blocks
        for block in self.blocks:
            h = block(h)
        
        # Final layer norm
        h = self.ln_f(h)
        
        # Output projection
        logits = self.output_proj(h)  # (batch, seq, vocab_size_with_mask)
        
        return logits


class TimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embeddings.
    """
    
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
        # MLP for timestep processing
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, t):
        """
        Args:
            t: (batch_size,) timesteps
        
        Returns:
            embeddings: (batch_size, dim)
        """
        # Create sinusoidal embeddings
        half_dim = self.dim // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Handle odd dimensions
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        
        # Process through MLP
        emb = self.mlp(emb)
        
        return emb


class TransformerBlock(nn.Module):
    """
    Standard transformer block with self-attention and feedforward.
    """
    
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # Layer norms
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        # Multi-head attention
        self.attn_qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.attn_out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Feedforward network
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_size)
        
        Returns:
            x: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Self-attention with residual
        residual = x
        x = self.ln1(x)
        
        # Compute Q, K, V
        qkv = self.attn_qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch, heads, seq, head_dim)
        out = out.transpose(1, 2).contiguous()  # (batch, seq, heads, head_dim)
        out = out.reshape(batch_size, seq_len, self.hidden_size)
        
        # Output projection and residual
        out = self.attn_out(out)
        x = residual + out
        
        # Feedforward with residual
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    import yaml
    
    config = {
        'model': {
            'vocab_size': 50257,
            'hidden_size': 768,
            'num_layers': 12,
            'num_heads': 12,
            'max_length': 1024,
            'dropout': 0.1
        }
    }
    
    # Convert to namespace for compatibility
    from types import SimpleNamespace
    config = SimpleNamespace(**{k: SimpleNamespace(**v) if isinstance(v, dict) else v 
                                for k, v in config.items()})
    
    model = D3PM(config)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    x = torch.randint(0, 50257, (batch_size, seq_len))
    t = torch.randint(0, 1000, (batch_size,))
    
    logits = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Timesteps shape: {t.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {50257 + 1})")
    
    assert logits.shape == (batch_size, seq_len, 50258), f"Wrong output shape: {logits.shape}"
    print("\n✅ D3PM model test passed!")
