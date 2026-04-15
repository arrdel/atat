"""
D3PM (Discrete Denoising Diffusion Probabilistic Models) baseline.

Implementation of Austin et al., NeurIPS 2021 for comparison with ATAT.
"""

from .d3pm_model import D3PM, count_parameters
from .diffusion import GaussianDiffusion

__all__ = ['D3PM', 'GaussianDiffusion', 'count_parameters']
__version__ = '1.0.0'
