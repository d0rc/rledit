"""
Evaluation metrics for the recursive text editor.

This module provides metrics for evaluating the performance of the text editor.
"""

import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.metrics.distance import edit_distance


class EvaluationMetrics:
    """
    Evaluation metrics for the text editor.
    
    This class provides methods for computing various evaluation metrics.
    """
    
    @staticmethod
    def bleu_score(reference, hypothesis):
        """
        Compute the BLEU score between a reference and a hypothesis.
        
        Args:
            reference: The reference text
            hypothesis: The hypothesis text
            
        Returns:
            bleu: The BLEU score
        """
        # Tokenize the texts
        reference_tokens = reference.split()
        hypothesis_tokens = hypothesis.split()
        
        # Compute the BLEU score
        smoothing = SmoothingFunction().method1
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothing)
        
        return bleu
    
    @staticmethod
    def edit_distance(reference, hypothesis):
        """
        Compute the edit distance between a reference and a hypothesis.
        
        Args:
            reference: The reference text
            hypothesis: The hypothesis text
            
        Returns:
            distance: The edit distance
        """
        # Compute the edit distance
        distance = edit_distance(reference, hypothesis)
        
        return distance
    
    @staticmethod
    def normalized_edit_distance(reference, hypothesis):
        """
        Compute the normalized edit distance between a reference and a hypothesis.
        
        Args:
            reference: The reference text
            hypothesis: The hypothesis text
            
        Returns:
            normalized_distance: The normalized edit distance
        """
        # Compute the edit distance
        distance = edit_distance(reference, hypothesis)
        
        # Normalize by the length of the reference
        normalized_distance = distance / max(len(reference), 1)
        
        return normalized_distance
    
    @staticmethod
    def exact_match(reference, hypothesis):
        """
        Check if the reference and hypothesis match exactly.
        
        Args:
            reference: The reference text
            hypothesis: The hypothesis text
            
        Returns:
            match: Whether the texts match exactly
        """
        return reference == hypothesis
    
    @staticmethod
    def token_accuracy(reference_tokens, hypothesis_tokens):
        """
        Compute the token-level accuracy between reference and hypothesis tokens.
        
        Args:
            reference_tokens: The reference tokens
            hypothesis_tokens: The hypothesis tokens
            
        Returns:
            accuracy: The token-level accuracy
        """
        # Compute the number of matching tokens
        num_matches = sum(1 for ref, hyp in zip(reference_tokens, hypothesis_tokens) if ref == hyp)
        
        # Compute the accuracy
        accuracy = num_matches / max(len(reference_tokens), 1)
        
        return accuracy
    
    @staticmethod
    def operation_accuracy(reference_operations, predicted_operations):
        """
        Compute the operation-level accuracy between reference and predicted operations.
        
        Args:
            reference_operations: The reference operations
            predicted_operations: The predicted operations
            
        Returns:
            accuracy: The operation-level accuracy
        """
        # Compute the number of matching operations
        num_matches = sum(1 for ref, pred in zip(reference_operations, predicted_operations) if ref == pred)
        
        # Compute the accuracy
        accuracy = num_matches / max(len(reference_operations), 1)
        
        return accuracy
    
    @staticmethod
    def convergence_rate(num_iterations, max_iterations):
        """
        Compute the convergence rate.
        
        Args:
            num_iterations: The number of iterations taken
            max_iterations: The maximum number of iterations
            
        Returns:
            rate: The convergence rate
        """
        # Compute the convergence rate
        rate = 1.0 - (num_iterations / max_iterations)
        
        return rate
    
    @staticmethod
    def quality_improvement(original_quality, edited_quality):
        """
        Compute the quality improvement.
        
        Args:
            original_quality: The quality of the original text
            edited_quality: The quality of the edited text
            
        Returns:
            improvement: The quality improvement
        """
        # Compute the quality improvement
        improvement = edited_quality - original_quality
        
        return improvement
    
    @staticmethod
    def efficiency(num_edits, quality_improvement):
        """
        Compute the edit efficiency.
        
        Args:
            num_edits: The number of edits made
            quality_improvement: The quality improvement
            
        Returns:
            efficiency: The edit efficiency
        """
        # Compute the efficiency
        efficiency = quality_improvement / max(num_edits, 1)
        
        return efficiency
    
    @staticmethod
    def compute_metrics(original_texts, edited_texts, reference_texts=None, edit_traces=None, max_iterations=5):
        """
        Compute various evaluation metrics.
        
        Args:
            original_texts: List of original texts
            edited_texts: List of edited texts
            reference_texts: List of reference texts (optional)
            edit_traces: List of edit traces (optional)
            max_iterations: Maximum number of iterations
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Compute BLEU score if reference texts are provided
        if reference_texts is not None:
            bleu_scores = [
                EvaluationMetrics.bleu_score(ref, hyp)
                for ref, hyp in zip(reference_texts, edited_texts)
            ]
            metrics["bleu_score"] = np.mean(bleu_scores)
        
        # Compute edit distance
        edit_distances = [
            EvaluationMetrics.edit_distance(orig, edit)
            for orig, edit in zip(original_texts, edited_texts)
        ]
        metrics["edit_distance"] = np.mean(edit_distances)
        
        # Compute normalized edit distance
        normalized_edit_distances = [
            EvaluationMetrics.normalized_edit_distance(orig, edit)
            for orig, edit in zip(original_texts, edited_texts)
        ]
        metrics["normalized_edit_distance"] = np.mean(normalized_edit_distances)
        
        # Compute exact match if reference texts are provided
        if reference_texts is not None:
            exact_matches = [
                EvaluationMetrics.exact_match(ref, hyp)
                for ref, hyp in zip(reference_texts, edited_texts)
            ]
            metrics["exact_match"] = np.mean(exact_matches)
        
        # Compute convergence rate if edit traces are provided
        if edit_traces is not None:
            convergence_rates = [
                EvaluationMetrics.convergence_rate(len(trace), max_iterations)
                for trace in edit_traces
            ]
            metrics["convergence_rate"] = np.mean(convergence_rates)
        
        return metrics
