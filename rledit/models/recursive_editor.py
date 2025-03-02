"""
Recursive editor module for the recursive text editor.

This module implements the recursive editor controller that applies edit operations
to input text until convergence or maximum iterations.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .edit_operations import EditOperation


class RecursiveEditor:
    """
    Recursive editor controller.
    
    This class handles the recursive editing process, applying edit operations
    to input text until convergence or maximum iterations.
    """
    
    def __init__(self, editor_model, tokenizer, max_iterations=5, convergence_threshold=0.95):
        """
        Initialize the recursive editor.
        
        Args:
            editor_model: The editor model
            tokenizer: The tokenizer
            max_iterations: Maximum number of iterations
            convergence_threshold: Threshold for convergence (fraction of KEEP operations)
        """
        self.editor_model = editor_model
        self.tokenizer = tokenizer
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
    
    def edit_until_convergence(self, texts, sample=False, temperature=1.0):
        """
        Edit texts until convergence or maximum iterations.
        
        Args:
            texts: List of input texts
            sample: Whether to sample operations or take the argmax
            temperature: Temperature for sampling
            
        Returns:
            edited_texts: The edited texts
            edit_traces: The edit traces for RL training
        """
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        batch_size = len(texts)
        current_texts = texts.copy()
        edit_traces = [[] for _ in range(batch_size)]
        
        for iteration in range(self.max_iterations):
            # Tokenize the current texts
            inputs = self.tokenizer(
                current_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=True,
            ).to(self.editor_model.device)
            
            # Predict edit operations
            with torch.no_grad():
                operations, replacements, splits, operation_probs, replacement_probs, split_probs = (
                    self.editor_model.predict_edit_operations(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        token_type_ids=inputs["token_type_ids"],
                        temperature=temperature,
                        sample=sample,
                    )
                )
            
            # Apply edit operations
            new_texts, batch_edit_traces = self._apply_edit_operations(
                current_texts,
                inputs["input_ids"],
                operations,
                replacements,
                splits,
                operation_probs,
                replacement_probs,
                split_probs,
            )
            
            # Update edit traces
            for i in range(batch_size):
                edit_traces[i].append(batch_edit_traces[i])
            
            # Check for convergence
            converged = self._check_convergence(operations, inputs["attention_mask"])
            if all(converged):
                break
            
            # Update current texts
            current_texts = new_texts
        
        if single_input:
            return current_texts[0], edit_traces[0]
        else:
            return current_texts, edit_traces
    
    def _apply_edit_operations(
        self,
        texts,
        input_ids,
        operations,
        replacements,
        splits,
        operation_probs,
        replacement_probs,
        split_probs,
    ):
        """
        Apply edit operations to the input texts.
        
        Args:
            texts: List of input texts
            input_ids: The input token IDs, shape (batch_size, seq_len)
            operations: The predicted operations, shape (batch_size, seq_len)
            replacements: The predicted replacement tokens, shape (batch_size, seq_len)
            splits: The predicted split tokens, shape (batch_size, seq_len, 2)
            operation_probs: The probabilities of the predicted operations, shape (batch_size, seq_len)
            replacement_probs: The probabilities of the predicted replacements, shape (batch_size, seq_len)
            split_probs: The probabilities of the predicted splits, shape (batch_size, seq_len, 2)
            
        Returns:
            edited_texts: The edited texts
            edit_traces: The edit traces for RL training
        """
        batch_size = len(texts)
        edited_texts = []
        edit_traces = []
        
        for i in range(batch_size):
            # Get the tokens for this example
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])
            
            # Apply edit operations
            edited_tokens = []
            trace = []
            
            for j, token in enumerate(tokens):
                # Skip special tokens
                if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                    continue
                
                # Get the operation for this token
                op = operations[i, j].item()
                op_prob = operation_probs[i, j].item()
                
                # Apply the operation
                if op == EditOperation.KEEP.value:
                    edited_tokens.append(token)
                    trace.append({
                        "operation": "KEEP",
                        "token": token,
                        "probability": op_prob,
                    })
                elif op == EditOperation.REMOVE.value:
                    trace.append({
                        "operation": "REMOVE",
                        "token": token,
                        "probability": op_prob,
                    })
                elif op == EditOperation.REPLACE.value:
                    replacement = self.tokenizer.convert_ids_to_tokens(replacements[i, j].item())
                    edited_tokens.append(replacement)
                    trace.append({
                        "operation": "REPLACE",
                        "token": token,
                        "replacement": replacement,
                        "probability": op_prob,
                        "replacement_probability": replacement_probs[i, j].item(),
                    })
                elif op == EditOperation.SPLIT.value:
                    split_tokens = [
                        self.tokenizer.convert_ids_to_tokens(splits[i, j, 0].item()),
                        self.tokenizer.convert_ids_to_tokens(splits[i, j, 1].item()),
                    ]
                    edited_tokens.extend(split_tokens)
                    trace.append({
                        "operation": "SPLIT",
                        "token": token,
                        "split_tokens": split_tokens,
                        "probability": op_prob,
                        "split_probabilities": [split_probs[i, j, 0].item(), split_probs[i, j, 1].item()],
                    })
            
            # Convert tokens back to text
            edited_text = self.tokenizer.convert_tokens_to_string(edited_tokens)
            edited_texts.append(edited_text)
            edit_traces.append(trace)
        
        return edited_texts, edit_traces
    
    def _check_convergence(self, operations, attention_mask):
        """
        Check if the editing process has converged.
        
        Args:
            operations: The predicted operations, shape (batch_size, seq_len)
            attention_mask: The attention mask, shape (batch_size, seq_len)
            
        Returns:
            converged: Boolean tensor indicating whether each example has converged
        """
        batch_size = operations.shape[0]
        converged = []
        
        for i in range(batch_size):
            # Count the number of non-KEEP operations
            non_keep = (operations[i] != EditOperation.KEEP.value).float()
            
            # Apply attention mask to ignore padding
            non_keep = non_keep * attention_mask[i]
            
            # Count the number of tokens
            num_tokens = attention_mask[i].sum().item()
            
            # Count the number of non-KEEP operations
            num_non_keep = non_keep.sum().item()
            
            # Check if the fraction of KEEP operations is above the threshold
            keep_fraction = 1.0 - (num_non_keep / num_tokens)
            converged.append(keep_fraction >= self.convergence_threshold)
        
        return converged
    
    def edit_with_sampling(self, texts, temperature=1.0):
        """
        Edit texts with sampling for RL training.
        
        Args:
            texts: List of input texts
            temperature: Temperature for sampling
            
        Returns:
            edited_texts: The edited texts
            edit_traces: The edit traces for RL training
        """
        return self.edit_until_convergence(texts, sample=True, temperature=temperature)
