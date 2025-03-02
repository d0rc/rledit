"""
Training module for the recursive text editor.

This module contains the implementation of the training pipeline,
including supervised pretraining and reinforcement learning.
"""

from .rl_trainer import RLTrainer
from .environment import (
    EditEnvironment,
    LanguageModelPerplexity,
    GrammarCorrectness,
    CombinedQualityEvaluator,
)
from .rewards import (
    RewardFunction,
    LanguageModelPerplexityReward,
    GrammarCorrectnessReward,
    IterationEfficiencyReward,
    EditDistanceReward,
    CombinedReward,
)

__all__ = [
    "RLTrainer",
    "EditEnvironment",
    "LanguageModelPerplexity",
    "GrammarCorrectness",
    "CombinedQualityEvaluator",
    "RewardFunction",
    "LanguageModelPerplexityReward",
    "GrammarCorrectnessReward",
    "IterationEfficiencyReward",
    "EditDistanceReward",
    "CombinedReward",
]
