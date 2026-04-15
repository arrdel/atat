"""
Noise schedules for discrete diffusion.
Adapted from: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
"""

import abc
import torch
import torch.nn as nn


class Noise(abc.ABC, nn.Module):
    """Base class for noise schedules."""

    def forward(self, t):
        """
        Returns total noise and rate of noise at timestep t.
        
        Args:
            t: timestep in [0, 1]
        
        Returns:
            (sigma, dsigma): total noise and noise rate
        """
        return self.total_noise(t), self.rate_noise(t)

    @abc.abstractmethod
    def rate_noise(self, t):
        """Rate of change of noise: g(t)."""
        pass

    @abc.abstractmethod
    def total_noise(self, t):
        """Total noise: ∫_0^t g(s) ds."""
        pass


class GeometricNoise(Noise):
    """
    Geometric noise schedule.
    Interpolates between sigma_min and sigma_max geometrically.
    """

    def __init__(self, sigma_min=1e-3, sigma_max=1.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_sigma_ratio = torch.log(torch.tensor(sigma_max / sigma_min))

    def rate_noise(self, t):
        """Rate: sigma_min * exp(t * log(sigma_max / sigma_min)) * log(sigma_max / sigma_min)."""
        return self.sigma_min * torch.exp(t * self.log_sigma_ratio) * self.log_sigma_ratio

    def total_noise(self, t):
        """Total: sigma_min * (exp(t * log(sigma_max / sigma_min)) - 1)."""
        return self.sigma_min * (torch.exp(t * self.log_sigma_ratio) - 1)


class LogLinearNoise(Noise, nn.Module):
    """
    Log-linear noise schedule.
    
    Total noise is -log(1 - (1 - eps) * t), so sigma = (1 - eps) * t
    This schedule ensures 1 - 1/e^{sigma(t)} interpolates between 0 and ~1.
    """

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        # Dummy parameter to register module
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(self, t):
        """Rate: (1 - eps) / (1 - (1 - eps) * t)."""
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        """Total: -log(1 - (1 - eps) * t)."""
        return -torch.log1p(-(1 - self.eps) * t)


def get_noise(config):
    """Factory function to create noise schedule based on config."""
    if hasattr(config, 'noise'):
        noise_config = config.noise
        noise_type = noise_config.type if hasattr(noise_config, 'type') else 'loglinear'
    else:
        noise_type = config.get('noise', {}).get('type', 'loglinear')
    
    if noise_type == "geometric":
        sigma_min = config.noise.get('sigma_min', 1e-3) if hasattr(config, 'noise') else 1e-3
        sigma_max = config.noise.get('sigma_max', 1.0) if hasattr(config, 'noise') else 1.0
        return GeometricNoise(sigma_min, sigma_max)
    elif noise_type == "loglinear":
        eps = config.noise.get('eps', 1e-3) if hasattr(config, 'noise') else 1e-3
        return LogLinearNoise(eps)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
