"""
Reward functions module for the recursive text editor.

This module implements various reward functions for reinforcement learning training.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod


class RewardFunction(ABC):
    """
    Abstract base class for reward functions.
    
    This class defines the interface for reward functions.
    """
    
    @abstractmethod
    def __call__(self, original_text, edited_text, edit_trace):
        """
        Compute the reward for an edit.
        
        Args:
            original_text: The original text
            edited_text: The edited text
            edit_trace: The edit trace
            
        Returns:
            reward: The reward for the edit
        """
        pass


class LanguageModelPerplexityReward(RewardFunction):
    """
    Language model perplexity reward.
    
    This reward function computes the improvement in language model perplexity.
    """
    
    def __init__(self, language_model_perplexity):
        """
        Initialize the language model perplexity reward.
        
        Args:
            language_model_perplexity: The language model perplexity evaluator
        """
        self.language_model_perplexity = language_model_perplexity
    
    def __call__(self, original_text, edited_text, edit_trace):
        """
        Compute the reward for an edit.
        
        Args:
            original_text: The original text
            edited_text: The edited text
            edit_trace: The edit trace
            
        Returns:
            reward: The reward for the edit
        """
        original_perplexity = self.language_model_perplexity(original_text)
        edited_perplexity = self.language_model_perplexity(edited_text)
        
        # Compute the improvement in perplexity
        # Note: Perplexity is negative, so a higher value is better
        improvement = edited_perplexity - original_perplexity
        
        return improvement


class GrammarCorrectnessReward(RewardFunction):
    """
    Grammar correctness reward.
    
    This reward function computes the improvement in grammar correctness.
    """
    
    def __init__(self, grammar_correctness):
        """
        Initialize the grammar correctness reward.
        
        Args:
            grammar_correctness: The grammar correctness evaluator
        """
        self.grammar_correctness = grammar_correctness
    
    def __call__(self, original_text, edited_text, edit_trace):
        """
        Compute the reward for an edit.
        
        Args:
            original_text: The original text
            edited_text: The edited text
            edit_trace: The edit trace
            
        Returns:
            reward: The reward for the edit
        """
        original_errors = self.grammar_correctness(original_text)
        edited_errors = self.grammar_correctness(edited_text)
        
        # Compute the improvement in grammar correctness
        # Note: Error count is negative, so a higher value is better
        improvement = edited_errors - original_errors
        
        return improvement


class IterationEfficiencyReward(RewardFunction):
    """
    Iteration efficiency reward.
    
    This reward function gives a bonus for using fewer iterations.
    """
    
    def __init__(self, max_iterations=5, efficiency_factor=0.1):
        """
        Initialize the iteration efficiency reward.
        
        Args:
            max_iterations: Maximum number of iterations
            efficiency_factor: Weight for the efficiency bonus
        """
        self.max_iterations = max_iterations
        self.efficiency_factor = efficiency_factor
    
    def __call__(self, original_text, edited_text, edit_trace):
        """
        Compute the reward for an edit.
        
        Args:
            original_text: The original text
            edited_text: The edited text
            edit_trace: The edit trace
            
        Returns:
            reward: The reward for the edit
        """
        num_iterations = len(edit_trace)
        
        # Compute the efficiency bonus
        efficiency_bonus = max(0, self.max_iterations - num_iterations) * self.efficiency_factor
        
        return efficiency_bonus


class EditDistanceReward(RewardFunction):
    """
    Edit distance reward.
    
    This reward function penalizes excessive changes.
    """
    
    def __init__(self, penalty_factor=0.01):
        """
        Initialize the edit distance reward.
        
        Args:
            penalty_factor: Weight for the edit distance penalty
        """
        self.penalty_factor = penalty_factor
    
    def __call__(self, original_text, edited_text, edit_trace):
        """
        Compute the reward for an edit.
        
        Args:
            original_text: The original text
            edited_text: The edited text
            edit_trace: The edit trace
            
        Returns:
            reward: The reward for the edit
        """
        # Count the number of non-KEEP operations
        num_edits = sum(1 for iteration in edit_trace for op in iteration if op["operation"] != "KEEP")
        
        # Compute the edit distance penalty
        edit_distance_penalty = -num_edits * self.penalty_factor
        
        return edit_distance_penalty


class CombinedReward(RewardFunction):
    """
    Combined reward.
    
    This reward function combines multiple reward functions.
    """
    
    def __init__(self, reward_functions=None, weights=None):
        """
        Initialize the combined reward.
        
        Args:
            reward_functions: Dictionary of reward functions
            weights: Dictionary of weights for each reward function
        """
        self.reward_functions = reward_functions or {}
        self.weights = weights or {}
    
    def add_reward_function(self, name, reward_function, weight=1.0):
        """
        Add a reward function.
        
        Args:
            name: The name of the reward function
            reward_function: The reward function
            weight: The weight of the reward function
        """
        self.reward_functions[name] = reward_function
        self.weights[name] = weight
    
    def __call__(self, original_text, edited_text, edit_trace):
        """
        Compute the reward for an edit.
        
        Args:
            original_text: The original text
            edited_text: The edited text
            edit_trace: The edit trace
            
        Returns:
            reward: The combined reward for the edit
        """
        reward = 0.0
        
        for name, reward_function in self.reward_functions.items():
            component_reward = reward_function(original_text, edited_text, edit_trace)
            weight = self.weights.get(name, 1.0)
            reward += component_reward * weight
        
        return reward
