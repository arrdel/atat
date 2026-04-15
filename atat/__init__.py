"""
ATAT: Adaptive Token Attention for Text Diffusion

Implementation of "Not All Tokens Are Equal: Importance-Aware Masking
for Discrete Diffusion Language Models" (ACL 2025).

Key Components:
    atat/               Core modules (importance, masking, curriculum, sampling)
    models/             ATAT-enhanced model architectures (ATATDiT)
    configs/            Configuration files
    utils/              Utility functions (frequency, dataloader)
    scripts/training/   Training scripts
    scripts/evaluation/ Evaluation scripts
"""

__version__ = "1.0.0"
__author__ = "Adele Chinda"

from atat.atat import (
    ImportanceEstimator,
    AdaptiveMaskingScheduler,
    CurriculumScheduler,
    UncertaintySampler,
)
from atat.models import ATATDiT

__all__ = [
    "ImportanceEstimator",
    "AdaptiveMaskingScheduler",
    "CurriculumScheduler",
    "UncertaintySampler",
    "ATATDiT",
]
