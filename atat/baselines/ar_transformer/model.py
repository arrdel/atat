"""
Autoregressive Transformer Baseline

Standard GPT-style autoregressive language model for comparison with ATAT.
Uses next-token prediction with cross-entropy loss.

Architecture matches ATAT's DiT backbone for fair comparison:
- Same hidden dimension (768)
- Same number of layers (12)
- Same attention heads (12)
- Same sequence length (1024)
- Same vocabulary size (50257)

The only difference is the training objective:
- AR: Next-token prediction p(x_t | x_<t)
- ATAT: Masked token prediction p(x_t | x_masked)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) from Su et al. 2021"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute sin/cos for efficiency
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to queries and keys."""
        seq_len = q.shape[2]
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE."""
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Rotary embeddings
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
        
        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
                1, 1, max_seq_len, max_seq_len
            ),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape  # batch, length, hidden_size
        
        # Compute Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape to (B, num_heads, L, head_dim)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        q, k = self.rope(q, k)
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        attn = attn.masked_fill(self.causal_mask[:, :, :L, :L] == 0, float("-inf"))
        
        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.proj(out)
        out = self.proj_dropout(out)
        
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""
    
    def __init__(self, hidden_size: int = 768, ffn_dim: int = 3072, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, num_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, ffn_dim, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class ARTransformer(nn.Module):
    """
    Autoregressive Transformer for language modeling.
    
    Architecture:
    - Token embeddings
    - N transformer blocks with causal attention
    - Layer norm
    - Output projection to vocabulary
    
    Training:
    - Cross-entropy loss on next-token prediction
    - Teacher forcing during training
    
    Args:
        vocab_size: Vocabulary size (default: 50257 for GPT-2)
        hidden_size: Hidden dimension (default: 768)
        num_layers: Number of transformer blocks (default: 12)
        num_heads: Number of attention heads (default: 12)
        ffn_dim: Feed-forward network dimension (default: 3072)
        max_seq_len: Maximum sequence length (default: 1024)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ffn_dim, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        
        # Output normalization and projection
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights between embedding and output projection (standard practice)
        self.head.weight = self.token_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
        
        Returns:
            logits: Logits of shape (batch_size, seq_len, vocab_size)
        """
        # Get embeddings
        x = self.token_emb(input_ids)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    def compute_loss(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute cross-entropy loss for next-token prediction.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
        
        Returns:
            loss: Scalar loss
            metrics: Dictionary of metrics (loss, perplexity)
        """
        # Forward pass
        logits = self.forward(input_ids)
        
        # Shift for next-token prediction
        # Input:  [x0, x1, x2, x3]
        # Target: [x1, x2, x3, x4]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        # Flatten and compute cross-entropy
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            reduction="mean",
        )
        
        # Compute perplexity
        perplexity = torch.exp(loss)
        
        metrics = {
            "loss": loss.item(),
            "perplexity": perplexity.item(),
        }
        
        return loss, metrics
    
    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            prompt: Token IDs of shape (batch_size, prompt_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (if not None)
            top_p: Top-p (nucleus) sampling (if not None)
        
        Returns:
            generated: Token IDs of shape (batch_size, prompt_len + max_new_tokens)
        """
        self.eval()
        generated = prompt
        
        for _ in range(max_new_tokens):
            # Get logits for next token
            logits = self.forward(generated)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            
            # Apply top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if we've generated enough tokens or hit EOS
            if generated.shape[1] >= self.max_seq_len:
                break
        
        return generated
    
    def num_parameters(self) -> int:
        """Count the number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = ARTransformer(
        vocab_size=50257,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        max_seq_len=1024,
    )
    
    print(f"Model parameters: {model.num_parameters():,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 512
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))
    
    logits = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test loss computation
    loss, metrics = model.compute_loss(input_ids)
    print(f"Loss: {loss.item():.4f}")
    print(f"Perplexity: {metrics['perplexity']:.4f}")
    
    # Test generation
    prompt = torch.randint(0, 50257, (1, 10))
    generated = model.generate(prompt, max_new_tokens=50, temperature=0.8)
    print(f"Generated shape: {generated.shape}")
