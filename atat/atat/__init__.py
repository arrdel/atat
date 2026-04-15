"""
ATAT – Adaptive Token Attention for Text Diffusion

Core modules implementing the method described in:
"Not All Tokens Are Equal: Importance-Aware Masking for Discrete
Diffusion Language Models" (ACL 2025).

Modules:
    ImportanceEstimator    – 2-layer MLP importance estimator (§3.1)
    AdaptiveMaskingScheduler – Balanced masking g_bal (§3.2)
    CurriculumScheduler   – 3-stage training curriculum (§3.3)
    UncertaintySampler    – Uncertainty-guided decoding (§3.4)
"""

from atat.atat.importance_estimator import ImportanceEstimator
from atat.atat.adaptive_masking import AdaptiveMaskingScheduler
from atat.atat.curriculum import CurriculumScheduler
from atat.atat.uncertainty_sampler import UncertaintySampler

__all__ = [
    "ImportanceEstimator",
    "AdaptiveMaskingScheduler",
    "CurriculumScheduler",
    "UncertaintySampler",
]
