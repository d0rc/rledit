"""
Environment module for the recursive text editor.

This module implements the environment for reinforcement learning training.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


class EditEnvironment:
    """
    Environment for reinforcement learning training.
    
    This class handles the computation of rewards for the RL training.
    """
    
    def __init__(
        self,
        quality_evaluator=None,
        max_iterations=5,
        efficiency_factor=0.1,
        reward_weights=None,
    ):
        """
        Initialize the edit environment.
        
        Args:
            quality_evaluator: Function or object that evaluates text quality
            max_iterations: Maximum number of iterations
            efficiency_factor: Weight for the efficiency bonus
            reward_weights: Dictionary of weights for different reward components
        """
        self.quality_evaluator = quality_evaluator
        self.max_iterations = max_iterations
        self.efficiency_factor = efficiency_factor
        self.reward_weights = reward_weights or {
            "fluency": 0.5,
            "grammar": 0.3,
            "efficiency": 0.2,
        }
    
    def compute_reward(self, original_text, edited_text, num_iterations):
        """
        Compute the reward for an edit.
        
        Args:
            original_text: The original text
            edited_text: The edited text
            num_iterations: The number of iterations taken
            
        Returns:
            reward: The reward for the edit
        """
        # Compute quality improvement
        if self.quality_evaluator is not None:
            original_quality = self.quality_evaluator(original_text)
            edited_quality = self.quality_evaluator(edited_text)
            quality_improvement = edited_quality - original_quality
        else:
            # Default to a simple length-based heuristic if no evaluator is provided
            if len(original_text) == 0:
                # Handle empty original text
                quality_improvement = 0.0 if len(edited_text) == 0 else 0.2
            else:
                quality_improvement = min(len(edited_text) / len(original_text) - 0.8, 0.2)
        
        # Compute efficiency bonus
        efficiency_bonus = max(0, self.max_iterations - num_iterations) * self.efficiency_factor
        
        # Combine rewards
        reward = (
            self.reward_weights.get("fluency", 0.5) * quality_improvement +
            self.reward_weights.get("efficiency", 0.2) * efficiency_bonus
        )
        
        return reward
    
    def compute_batch_rewards(self, original_texts, edited_texts, edit_traces):
        """
        Compute rewards for a batch of edits.
        
        Args:
            original_texts: List of original texts
            edited_texts: List of edited texts
            edit_traces: List of edit traces
            
        Returns:
            rewards: List of rewards
        """
        rewards = []
        
        for original, edited, trace in zip(original_texts, edited_texts, edit_traces):
            num_iterations = len(trace)
            reward = self.compute_reward(original, edited, num_iterations)
            rewards.append(reward)
        
        return rewards


class LanguageModelPerplexity:
    """
    Language model perplexity evaluator.
    
    This class computes the perplexity of text using a language model.
    Lower perplexity indicates better fluency.
    """
    
    def __init__(self, model_name="gpt2", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the language model perplexity evaluator.
        
        Args:
            model_name: The name of the language model
            device: The device to run the model on
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device
    
    def __call__(self, text):
        """
        Compute the perplexity of text.
        
        Args:
            text: The text to evaluate
            
        Returns:
            quality: The quality score (negative perplexity)
        """
        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Compute the perplexity
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        # Return negative perplexity as the quality score (higher is better)
        return -perplexity


class GrammarCorrectness:
    """
    Grammar correctness evaluator.
    
    This class evaluates the grammar correctness of text using a grammar checker.
    """
    
    def __init__(self):
        """Initialize the grammar correctness evaluator."""
        try:
            import language_tool_python
            self.tool = language_tool_python.LanguageTool("en-US")
        except ImportError:
            print("Warning: language-tool-python not installed. Using dummy grammar checker.")
            self.tool = None
    
    def __call__(self, text):
        """
        Evaluate the grammar correctness of text.
        
        Args:
            text: The text to evaluate
            
        Returns:
            quality: The quality score (negative number of errors)
        """
        if self.tool is None:
            # Dummy implementation
            return 0.0
        
        # Count the number of grammar errors
        matches = self.tool.check(text)
        num_errors = len(matches)
        
        # Return negative number of errors as the quality score (higher is better)
        return -num_errors


class CombinedQualityEvaluator:
    """
    Combined quality evaluator.
    
    This class combines multiple quality evaluators.
    """
    
    def __init__(self, evaluators=None, weights=None):
        """
        Initialize the combined quality evaluator.
        
        Args:
            evaluators: Dictionary of evaluators
            weights: Dictionary of weights for each evaluator
        """
        self.evaluators = evaluators or {}
        self.weights = weights or {}
    
    def add_evaluator(self, name, evaluator, weight=1.0):
        """
        Add an evaluator.
        
        Args:
            name: The name of the evaluator
            evaluator: The evaluator function or object
            weight: The weight of the evaluator
        """
        self.evaluators[name] = evaluator
        self.weights[name] = weight
    
    def __call__(self, text):
        """
        Evaluate the quality of text.
        
        Args:
            text: The text to evaluate
            
        Returns:
            quality: The combined quality score
        """
        quality = 0.0
        
        for name, evaluator in self.evaluators.items():
            score = evaluator(text)
            weight = self.weights.get(name, 1.0)
            quality += score * weight
        
        return quality
