"""
Edit operations module for the recursive text editor.

This module defines the edit operations and the edit operation head.
"""

import enum
import torch
import torch.nn as nn
import torch.nn.functional as F


class EditOperation(enum.Enum):
    """Enum for the edit operations."""
    KEEP = 0
    REMOVE = 1
    SPLIT = 2
    REPLACE = 3


class EditOperationHead(nn.Module):
    """
    Head for predicting edit operations.
    
    This module predicts the edit operations for each token in the input sequence.
    It consists of three components:
    1. Operation classifier: Predicts the operation type (KEEP, REMOVE, SPLIT, REPLACE)
    2. Replacement classifier: Predicts the replacement token for REPLACE operations
    3. Split classifier: Predicts the two tokens for SPLIT operations
    """
    
    def __init__(self, hidden_size, vocab_size):
        """
        Initialize the edit operation head.
        
        Args:
            hidden_size: The size of the hidden states from the encoder
            vocab_size: The size of the vocabulary
        """
        super().__init__()
        
        # Four operation logits: KEEP, REMOVE, SPLIT, REPLACE
        self.operation_classifier = nn.Linear(hidden_size, len(EditOperation))
        
        # Token replacement logits
        self.replacement_classifier = nn.Linear(hidden_size, vocab_size)
        
        # For split operations (simplified version)
        self.split_classifier = nn.Linear(hidden_size, vocab_size * 2)
        
    def forward(self, hidden_states):
        """
        Forward pass of the edit operation head.
        
        Args:
            hidden_states: The hidden states from the encoder, shape (batch_size, seq_len, hidden_size)
            
        Returns:
            operation_logits: Logits for the operation type, shape (batch_size, seq_len, num_operations)
            replacement_logits: Logits for the replacement token, shape (batch_size, seq_len, vocab_size)
            split_logits: Logits for the split tokens, shape (batch_size, seq_len, vocab_size * 2)
        """
        operation_logits = self.operation_classifier(hidden_states)
        replacement_logits = self.replacement_classifier(hidden_states)
        
        # Split logits are reshaped to (batch_size, seq_len, 2, vocab_size) later
        split_logits = self.split_classifier(hidden_states)
        
        return operation_logits, replacement_logits, split_logits
    
    def sample_operations(self, hidden_states, temperature=1.0):
        """
        Sample edit operations from the predicted logits.
        
        Args:
            hidden_states: The hidden states from the encoder, shape (batch_size, seq_len, hidden_size)
            temperature: Temperature for sampling, higher values make the distribution more uniform
            
        Returns:
            operations: The sampled operations, shape (batch_size, seq_len)
            replacements: The sampled replacement tokens, shape (batch_size, seq_len)
            splits: The sampled split tokens, shape (batch_size, seq_len, 2)
            operation_probs: The probabilities of the sampled operations, shape (batch_size, seq_len)
            replacement_probs: The probabilities of the sampled replacements, shape (batch_size, seq_len)
            split_probs: The probabilities of the sampled splits, shape (batch_size, seq_len, 2)
        """
        operation_logits, replacement_logits, split_logits = self.forward(hidden_states)
        
        # Scale logits by temperature
        operation_logits = operation_logits / temperature
        replacement_logits = replacement_logits / temperature
        split_logits = split_logits / temperature
        
        # Convert logits to probabilities
        operation_probs = F.softmax(operation_logits, dim=-1)
        replacement_probs = F.softmax(replacement_logits, dim=-1)
        
        # Reshape split logits to (batch_size, seq_len, 2, vocab_size)
        batch_size, seq_len, _ = split_logits.shape
        split_logits = split_logits.view(batch_size, seq_len, 2, -1)
        split_probs = F.softmax(split_logits, dim=-1)
        
        # Sample operations
        operation_dist = torch.distributions.Categorical(operation_probs)
        operations = operation_dist.sample()
        
        # Sample replacements
        replacement_dist = torch.distributions.Categorical(replacement_probs)
        replacements = replacement_dist.sample()
        
        # Sample splits
        split_dist = torch.distributions.Categorical(split_probs)
        splits = split_dist.sample()
        
        # Get probabilities of sampled operations
        operation_probs = torch.gather(
            operation_probs, -1, operations.unsqueeze(-1)
        ).squeeze(-1)
        
        # Get probabilities of sampled replacements
        replacement_probs = torch.gather(
            replacement_probs, -1, replacements.unsqueeze(-1)
        ).squeeze(-1)
        
        # Get probabilities of sampled splits
        split_probs = torch.gather(
            split_probs, -1, splits.unsqueeze(-1)
        ).squeeze(-1)
        
        return (
            operations, 
            replacements, 
            splits, 
            operation_probs, 
            replacement_probs, 
            split_probs
        )
