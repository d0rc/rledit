"""
Dataset module for the recursive text editor.

This module implements the dataset class for handling edit data.
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from ..models.edit_operations import EditOperation


class EditDataset(Dataset):
    """
    Dataset for edit data.
    
    This dataset handles pairs of original and edited texts for supervised pretraining.
    """
    
    def __init__(self, original_texts, edited_texts, tokenizer, max_length=512):
        """
        Initialize the edit dataset.
        
        Args:
            original_texts: List of original texts
            edited_texts: List of edited texts
            tokenizer: The tokenizer
            max_length: Maximum sequence length
        """
        self.original_texts = original_texts
        self.edited_texts = edited_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        assert len(original_texts) == len(edited_texts), "Original and edited texts must have the same length"
    
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.original_texts)
    
    def __getitem__(self, idx):
        """
        Get an example from the dataset.
        
        Args:
            idx: The index of the example
            
        Returns:
            A dictionary containing the example
        """
        original_text = self.original_texts[idx]
        edited_text = self.edited_texts[idx]
        
        # Tokenize the original text
        original_encoding = self.tokenizer(
            original_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Extract the edit operations
        edit_labels = self._extract_edit_operations(original_text, edited_text)
        
        # Create the example
        example = {
            "original_text": original_text,
            "edited_text": edited_text,
            "input_ids": original_encoding["input_ids"].squeeze(0),
            "attention_mask": original_encoding["attention_mask"].squeeze(0),
            "token_type_ids": original_encoding["token_type_ids"].squeeze(0) if "token_type_ids" in original_encoding else None,
            "edit_labels": edit_labels,
        }
        
        return example
    
    def _extract_edit_operations(self, original_text, edited_text):
        """
        Extract edit operations from original and edited texts.
        
        This is a simplified implementation that uses a dynamic programming approach
        to find the minimum edit distance and extract the edit operations.
        
        Args:
            original_text: The original text
            edited_text: The edited text
            
        Returns:
            edit_labels: Tensor of edit labels, shape (seq_len, 4)
                The first dimension is the operation type (KEEP, REMOVE, SPLIT, REPLACE)
                The second dimension is the replacement token ID (for REPLACE operations)
                The third and fourth dimensions are the split token IDs (for SPLIT operations)
        """
        # Tokenize the texts
        original_tokens = self.tokenizer.tokenize(original_text)
        edited_tokens = self.tokenizer.tokenize(edited_text)
        
        # Initialize the edit labels
        edit_labels = torch.zeros(self.max_length, 4, dtype=torch.long)
        
        # Compute the edit distance matrix
        m, n = len(original_tokens), len(edited_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize the first row and column
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if original_tokens[i - 1] == edited_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,  # Remove
                        dp[i][j - 1] + 1,  # Insert
                        dp[i - 1][j - 1] + 1,  # Replace
                    )
        
        # Backtrack to find the edit operations
        i, j = m, n
        edit_ops = []
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and original_tokens[i - 1] == edited_tokens[j - 1]:
                # Keep
                edit_ops.append((EditOperation.KEEP.value, None, None, None))
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                # Replace
                replacement_id = self.tokenizer.convert_tokens_to_ids(edited_tokens[j - 1])
                edit_ops.append((EditOperation.REPLACE.value, replacement_id, None, None))
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                # Remove
                edit_ops.append((EditOperation.REMOVE.value, None, None, None))
                i -= 1
            elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
                # Insert (treated as split for simplicity)
                if i > 0:
                    # Split the current token
                    split_id_1 = self.tokenizer.convert_tokens_to_ids(edited_tokens[j - 1])
                    split_id_2 = self.tokenizer.convert_tokens_to_ids(edited_tokens[j - 2]) if j > 1 else 0
                    edit_ops.append((EditOperation.SPLIT.value, None, split_id_1, split_id_2))
                    i -= 1
                    j -= 2
                else:
                    # Insert at the beginning (not handled in this simplified implementation)
                    j -= 1
        
        # Reverse the edit operations
        edit_ops.reverse()
        
        # Convert to tensor
        for i, (op, replacement_id, split_id_1, split_id_2) in enumerate(edit_ops):
            if i >= self.max_length:
                break
            edit_labels[i, 0] = op
            if replacement_id is not None:
                edit_labels[i, 1] = replacement_id
            if split_id_1 is not None:
                edit_labels[i, 2] = split_id_1
            if split_id_2 is not None:
                edit_labels[i, 3] = split_id_2
        
        return edit_labels


class EditCollator:
    """
    Collator for edit data.
    
    This collator handles batching of edit data.
    """
    
    def __init__(self, tokenizer):
        """
        Initialize the edit collator.
        
        Args:
            tokenizer: The tokenizer
        """
        self.tokenizer = tokenizer
    
    def __call__(self, examples):
        """
        Collate a batch of examples.
        
        Args:
            examples: List of examples
            
        Returns:
            A dictionary containing the batch
        """
        # Extract the fields
        original_texts = [example["original_text"] for example in examples]
        edited_texts = [example["edited_text"] for example in examples]
        input_ids = torch.stack([example["input_ids"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        edit_labels = torch.stack([example["edit_labels"] for example in examples])
        
        # Create the batch
        batch = {
            "original_texts": original_texts,
            "edited_texts": edited_texts,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "edit_labels": edit_labels,
        }
        
        # Add token type IDs if available
        if examples[0]["token_type_ids"] is not None:
            token_type_ids = torch.stack([example["token_type_ids"] for example in examples])
            batch["token_type_ids"] = token_type_ids
        
        return batch
