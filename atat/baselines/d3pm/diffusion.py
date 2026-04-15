"""
Discrete diffusion process for D3PM.

Implements the absorbing state diffusion process from Austin et al., NeurIPS 2021.
"""

import torch
import torch.nn.functional as F
import numpy as np


class GaussianDiffusion:
    """
    Discrete diffusion with absorbing state.
    
    All tokens gradually transition to a single mask token over the diffusion process.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Diffusion parameters
        if hasattr(config, 'diffusion'):
            diff_config = config.diffusion
            self.num_timesteps = diff_config.num_timesteps if hasattr(diff_config, 'num_timesteps') else 1000
            self.schedule = diff_config.schedule if hasattr(diff_config, 'schedule') else 'cosine'
            self.objective = diff_config.objective if hasattr(diff_config, 'objective') else 'vlb'
        else:
            self.num_timesteps = config.get('num_timesteps', 1000)
            self.schedule = config.get('schedule', 'cosine')
            self.objective = config.get('objective', 'vlb')
        
        # Create noise schedule
        self.betas = self._get_noise_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        
        # For sampling
        self.alphas_cumprod_prev = np.concatenate([[1.0], self.alphas_cumprod[:-1]])
        
    def _get_noise_schedule(self):
        """Create noise schedule."""
        if self.schedule == 'linear':
            # Linear schedule from Ho et al.
            scale = 1000 / self.num_timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return np.linspace(beta_start, beta_end, self.num_timesteps, dtype=np.float64)
        
        elif self.schedule == 'cosine':
            # Cosine schedule from Nichol & Dhariwal
            s = 0.008
            steps = self.num_timesteps + 1
            x = np.linspace(0, self.num_timesteps, steps)
            alphas_cumprod = np.cos(((x / self.num_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return np.clip(betas, 0.0001, 0.9999)
        
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
    
    def q_sample(self, x_0, t, noise=None):
        """
        Sample from q(x_t | x_0) - forward diffusion process.
        
        Args:
            x_0: (batch, seq_len) clean tokens
            t: (batch,) timesteps
            noise: optional pre-generated noise
        
        Returns:
            x_t: (batch, seq_len) noisy tokens
        """
        batch_size, seq_len = x_0.shape
        device = x_0.device
        
        # Get alpha values for this timestep
        alphas_t = torch.from_numpy(self.alphas_cumprod[t.cpu().numpy()]).to(device).float()
        alphas_t = alphas_t.reshape(-1, 1)  # (batch, 1)
        
        # Probability of keeping original token
        keep_prob = alphas_t
        
        # Mask token ID (vocab_size)
        mask_token = x_0.max() + 1  # Assumes mask token is vocab_size
        
        # Sample which tokens to mask
        if noise is None:
            noise = torch.rand(batch_size, seq_len, device=device)
        
        # Keep original token with probability alphas_t, otherwise mask
        mask = noise < keep_prob
        x_t = torch.where(mask, x_0, torch.full_like(x_0, mask_token))
        
        return x_t
    
    def p_mean(self, model, x_t, t):
        """
        Predict x_0 from x_t.
        
        Args:
            model: D3PM model
            x_t: (batch, seq_len) noisy tokens
            t: (batch,) timesteps
        
        Returns:
            logits: (batch, seq_len, vocab_size) predicted token logits
        """
        # Get model predictions
        logits = model(x_t, t)  # (batch, seq_len, vocab_size)
        return logits
    
    def p_sample(self, model, x_t, t):
        """
        Sample from p(x_{t-1} | x_t).
        
        Args:
            model: D3PM model
            x_t: (batch, seq_len) noisy tokens
            t: (batch,) timesteps
        
        Returns:
            x_{t-1}: (batch, seq_len) less noisy tokens
        """
        batch_size, seq_len = x_t.shape
        device = x_t.device
        
        # Get model predictions
        logits = self.p_mean(model, x_t, t)  # (batch, seq_len, vocab_size)
        
        # Sample from categorical distribution
        probs = F.softmax(logits, dim=-1)
        x_pred = torch.multinomial(probs.reshape(-1, probs.size(-1)), num_samples=1)
        x_pred = x_pred.reshape(batch_size, seq_len)
        
        # If t=0, return prediction directly
        is_zero = (t == 0).reshape(-1, 1)
        
        # Otherwise, apply reverse transition
        if not is_zero.all():
            # Get previous timestep alpha
            t_prev = torch.clamp(t - 1, min=0)
            alphas_t = torch.from_numpy(self.alphas_cumprod[t.cpu().numpy()]).to(device).float()
            alphas_t_prev = torch.from_numpy(self.alphas_cumprod[t_prev.cpu().numpy()]).to(device).float()
            
            # Transition probability
            keep_prob = (alphas_t_prev / alphas_t).reshape(-1, 1)
            
            # Sample transition
            noise = torch.rand(batch_size, seq_len, device=device)
            mask_token = logits.size(-1) - 1
            
            # Keep prediction with probability, otherwise stay masked
            keep = noise < keep_prob
            x_next = torch.where(keep, x_pred, torch.full_like(x_pred, mask_token))
            
            # At t=0, always use prediction
            x_next = torch.where(is_zero, x_pred, x_next)
        else:
            x_next = x_pred
        
        return x_next
    
    def compute_loss(self, model, x_0, t=None):
        """
        Compute diffusion loss.
        
        Args:
            model: D3PM model
            x_0: (batch, seq_len) clean tokens
            t: optional (batch,) timesteps (if None, sample uniformly)
        
        Returns:
            loss: scalar loss value
        """
        batch_size, seq_len = x_0.shape
        device = x_0.device
        
        # Sample timesteps uniformly if not provided
        if t is None:
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        # Forward diffusion: q(x_t | x_0)
        x_t = self.q_sample(x_0, t)
        
        # Model prediction
        logits = model(x_t, t)  # (batch, seq_len, vocab_size)
        
        # Compute cross-entropy loss
        # Only compute loss on originally unmasked positions
        mask_token = logits.size(-1) - 1
        loss_mask = (x_t != mask_token).float()  # Don't compute loss on mask tokens
        
        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            x_0.reshape(-1),
            reduction='none'
        )
        loss = loss.reshape(batch_size, seq_len)
        
        # Apply mask and average
        loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)
        
        return loss
    
    @torch.no_grad()
    def sample(self, model, shape, device='cuda'):
        """
        Sample from the model.
        
        Args:
            model: D3PM model
            shape: (batch_size, seq_len) output shape
            device: device to sample on
        
        Returns:
            samples: (batch_size, seq_len) generated tokens
        """
        batch_size, seq_len = shape
        
        # Start from all mask tokens
        vocab_size = model.vocab_size_with_mask
        mask_token = vocab_size - 1
        x_t = torch.full((batch_size, seq_len), mask_token, device=device, dtype=torch.long)
        
        # Reverse diffusion process
        for t_idx in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t)
        
        return x_t
    
    @torch.no_grad()
    def sample_progressive(self, model, shape, device='cuda', save_every=100):
        """
        Sample with intermediate steps saved.
        
        Args:
            model: D3PM model
            shape: (batch_size, seq_len)
            device: device
            save_every: save intermediate results every N steps
        
        Returns:
            samples: final samples
            intermediates: list of intermediate samples
        """
        batch_size, seq_len = shape
        intermediates = []
        
        # Start from all mask tokens
        vocab_size = model.vocab_size_with_mask
        mask_token = vocab_size - 1
        x_t = torch.full((batch_size, seq_len), mask_token, device=device, dtype=torch.long)
        
        # Reverse diffusion
        for t_idx in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t)
            
            if t_idx % save_every == 0:
                intermediates.append(x_t.clone())
        
        return x_t, intermediates


if __name__ == "__main__":
    # Test diffusion process
    from types import SimpleNamespace
    
    config = SimpleNamespace(
        diffusion=SimpleNamespace(
            num_timesteps=1000,
            schedule='cosine',
            objective='vlb'
        )
    )
    
    diffusion = GaussianDiffusion(config)
    print(f"✓ Created diffusion with {diffusion.num_timesteps} timesteps")
    print(f"✓ Schedule: {diffusion.schedule}")
    print(f"✓ Beta range: [{diffusion.betas.min():.4f}, {diffusion.betas.max():.4f}]")
    print(f"✓ Alpha_cumprod range: [{diffusion.alphas_cumprod.min():.4f}, {diffusion.alphas_cumprod.max():.4f}]")
    
    # Test forward process
    batch_size = 2
    seq_len = 128
    vocab_size = 50257
    
    x_0 = torch.randint(0, vocab_size, (batch_size, seq_len))
    t = torch.randint(0, 1000, (batch_size,))
    
    x_t = diffusion.q_sample(x_0, t)
    print(f"\n✓ Forward diffusion:")
    print(f"  Input shape: {x_0.shape}")
    print(f"  Noisy shape: {x_t.shape}")
    print(f"  Num masked: {(x_t == vocab_size).sum().item()} / {batch_size * seq_len}")
    
    print("\n✅ Diffusion process test passed!")
