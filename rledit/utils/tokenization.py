"""
Tokenization utilities for the recursive text editor.

This module provides utilities for handling tokenization.
"""

import torch
from transformers import AutoTokenizer


class TokenizationUtils:
    """
    Utilities for handling tokenization.
    
    This class provides methods for handling tokenization and token-level operations.
    """
    
    def __init__(self, tokenizer):
        """
        Initialize the tokenization utilities.
        
        Args:
            tokenizer: The tokenizer
        """
        self.tokenizer = tokenizer
    
    def get_token_to_char_mappings(self, text):
        """
        Get the token to character mappings for a text.
        
        Args:
            text: The text
            
        Returns:
            token_to_char: Dictionary mapping token indices to character spans
        """
        # Tokenize the text with offsets
        encoding = self.tokenizer(text, return_offsets_mapping=True)
        
        # Extract the offsets
        offsets = encoding["offset_mapping"]
        
        # Create the token to character mappings
        token_to_char = {}
        for i, (start, end) in enumerate(offsets):
            token_to_char[i] = (start, end)
        
        return token_to_char
    
    def get_char_to_token_mappings(self, text):
        """
        Get the character to token mappings for a text.
        
        Args:
            text: The text
            
        Returns:
            char_to_token: Dictionary mapping character indices to token indices
        """
        # Tokenize the text with offsets
        encoding = self.tokenizer(text, return_offsets_mapping=True)
        
        # Extract the offsets
        offsets = encoding["offset_mapping"]
        
        # Create the character to token mappings
        char_to_token = {}
        for i, (start, end) in enumerate(offsets):
            for j in range(start, end):
                char_to_token[j] = i
        
        return char_to_token
    
    def get_word_to_token_mappings(self, text):
        """
        Get the word to token mappings for a text.
        
        Args:
            text: The text
            
        Returns:
            word_to_tokens: Dictionary mapping word indices to token indices
        """
        # Tokenize the text with offsets
        encoding = self.tokenizer(text, return_offsets_mapping=True)
        
        # Extract the offsets
        offsets = encoding["offset_mapping"]
        
        # Split the text into words
        words = text.split()
        
        # Create the word to token mappings
        word_to_tokens = {}
        word_index = 0
        word_start = 0
        
        for i, (start, end) in enumerate(offsets):
            # Skip special tokens
            if start == 0 and end == 0:
                continue
            
            # Check if this token is part of the current word
            if start >= word_start:
                # Find the word that contains this token
                while word_index < len(words):
                    word = words[word_index]
                    word_end = word_start + len(word)
                    
                    if start < word_end:
                        # This token is part of the current word
                        if word_index not in word_to_tokens:
                            word_to_tokens[word_index] = []
                        word_to_tokens[word_index].append(i)
                        break
                    
                    # Move to the next word
                    word_index += 1
                    word_start = word_end + 1
        
        return word_to_tokens
    
    def get_token_to_word_mappings(self, text):
        """
        Get the token to word mappings for a text.
        
        Args:
            text: The text
            
        Returns:
            token_to_word: Dictionary mapping token indices to word indices
        """
        # Get the word to token mappings
        word_to_tokens = self.get_word_to_token_mappings(text)
        
        # Create the token to word mappings
        token_to_word = {}
        for word_index, token_indices in word_to_tokens.items():
            for token_index in token_indices:
                token_to_word[token_index] = word_index
        
        return token_to_word
    
    def apply_token_level_edits(self, text, operations, replacements=None, splits=None):
        """
        Apply token-level edits to a text.
        
        Args:
            text: The text
            operations: The operations to apply, shape (seq_len,)
            replacements: The replacement tokens, shape (seq_len,)
            splits: The split tokens, shape (seq_len, 2)
            
        Returns:
            edited_text: The edited text
        """
        # Tokenize the text
        encoding = self.tokenizer(text, return_tensors="pt")
        input_ids = encoding["input_ids"][0]
        
        # Apply the operations
        edited_ids = []
        for i, op in enumerate(operations):
            # Skip special tokens
            if i >= len(input_ids) or input_ids[i] in [
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.pad_token_id,
            ]:
                continue
            
            # Apply the operation
            if op == 0:  # KEEP
                edited_ids.append(input_ids[i].item())
            elif op == 1:  # REMOVE
                pass
            elif op == 2:  # SPLIT
                if splits is not None:
                    edited_ids.append(splits[i, 0].item())
                    edited_ids.append(splits[i, 1].item())
            elif op == 3:  # REPLACE
                if replacements is not None:
                    edited_ids.append(replacements[i].item())
        
        # Convert the edited IDs back to text
        edited_text = self.tokenizer.decode(edited_ids, skip_special_tokens=True)
        
        return edited_text
