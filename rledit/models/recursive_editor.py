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
    
    def __init__(self, editor_model, tokenizer, max_iterations=5, convergence_threshold=0.95, cache_size=1000):
        """
        Initialize the recursive editor.
        
        Args:
            editor_model: The editor model
            tokenizer: The tokenizer
            max_iterations: Maximum number of iterations
            convergence_threshold: Threshold for convergence (fraction of KEEP operations)
            cache_size: Maximum number of entries in the tokenization cache
        """
        self.editor_model = editor_model
        self.tokenizer = tokenizer
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.token_cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _cached_tokenize(self, texts, **kwargs):
        """
        Tokenize texts with caching.
        
        Args:
            texts: List of input texts
            **kwargs: Additional arguments for the tokenizer
            
        Returns:
            inputs: Tokenized inputs
        """
        batch_size = len(texts)
        cache_keys = []
        cached_results = []
        texts_to_tokenize = []
        indices_to_tokenize = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = (text, frozenset(kwargs.items()))
            cache_keys.append(cache_key)
            
            if cache_key in self.token_cache:
                # Cache hit
                cached_results.append(self.token_cache[cache_key])
                self.cache_hits += 1
            else:
                # Cache miss
                texts_to_tokenize.append(text)
                indices_to_tokenize.append(i)
                self.cache_misses += 1
        
        # If all texts were in cache, combine and return
        if not texts_to_tokenize:
            return self._combine_tokenized_inputs(cached_results, batch_size)
        
        # Tokenize texts not in cache
        tokenized_inputs = self.tokenizer(
            texts_to_tokenize,
            return_tensors="pt",
            padding=True,
            **kwargs
        ).to(self.editor_model.device)
        
        # Store in cache
        for i, idx in enumerate(indices_to_tokenize):
            # For single-item batches, need to ensure we get a proper subset
            if len(texts_to_tokenize) == 1:
                single_input = {k: v for k, v in tokenized_inputs.items()}
            else:
                # Select the specific item from the batch
                single_input = {
                    k: v[i:i+1] for k, v in tokenized_inputs.items()
                }
            
            # Cache the result
            self.token_cache[cache_keys[idx]] = single_input
            cached_results.append(single_input)
            
            # Manage cache size
            if len(self.token_cache) > self.cache_size:
                # Simple LRU-like behavior: remove a random item
                # In a more advanced implementation, this would use a proper LRU algorithm
                self.token_cache.pop(next(iter(self.token_cache)))
        
        # Combine cached and newly tokenized inputs
        return self._combine_tokenized_inputs(cached_results, batch_size)
    
    def _combine_tokenized_inputs(self, tokenized_inputs, batch_size):
        """
        Combine multiple tokenized inputs into a single batch.
        
        Args:
            tokenized_inputs: List of tokenized inputs
            batch_size: The total batch size
            
        Returns:
            combined_inputs: Combined tokenized inputs
        """
        # Initialize the result with the first input
        result = {k: [] for k in tokenized_inputs[0].keys()}
        
        # Combine all inputs
        for inputs in tokenized_inputs:
            for k, v in inputs.items():
                result[k].append(v)
        
        # Concatenate tensors
        for k, v in result.items():
            result[k] = torch.cat(v, dim=0)
        
        return result
    
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
        
        # Track which examples have converged
        converged_mask = [False] * batch_size
        final_texts = current_texts.copy()
        
        for iteration in range(self.max_iterations):
            # Filter out converged examples
            active_indices = [i for i, converged in enumerate(converged_mask) if not converged]
            
            # If all examples have converged, we're done
            if not active_indices:
                break
                
            # Process only non-converged examples
            active_texts = [current_texts[i] for i in active_indices]
            
            # Use cached tokenization
            inputs = self._cached_tokenize(
                active_texts,
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=True,
            )
            
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
            new_active_texts, active_batch_edit_traces = self._apply_edit_operations(
                active_texts,
                inputs["input_ids"],
                operations,
                replacements,
                splits,
                operation_probs,
                replacement_probs,
                split_probs,
            )
            
            # Check for convergence of active examples
            active_converged = self._check_convergence(operations, inputs["attention_mask"])
            
            # Update texts, traces, and convergence status for active examples
            for idx, active_idx in enumerate(active_indices):
                # Update edit traces
                edit_traces[active_idx].append(active_batch_edit_traces[idx])
                
                # Update current texts
                current_texts[active_idx] = new_active_texts[idx]
                
                # Update final texts
                final_texts[active_idx] = new_active_texts[idx]
                
                # Update convergence status
                if active_converged[idx]:
                    converged_mask[active_idx] = True
        
        if single_input:
            return final_texts[0], edit_traces[0]
        else:
            return final_texts, edit_traces
    
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
        # Vectorized implementation
        # Count the number of non-KEEP operations for each example
        non_keep = (operations != EditOperation.KEEP.value).float() * attention_mask
        
        # Count the number of tokens for each example
        num_tokens = attention_mask.sum(dim=1)
        
        # Count the number of non-KEEP operations for each example
        num_non_keep = non_keep.sum(dim=1)
        
        # Calculate the fraction of KEEP operations for each example
        keep_fraction = 1.0 - (num_non_keep / num_tokens)
        
        # Check if the fraction of KEEP operations is above the threshold
        converged = (keep_fraction >= self.convergence_threshold).tolist()
        
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
    
    def get_cache_stats(self):
        """
        Get statistics about the tokenization cache.
        
        Returns:
            stats: Dictionary containing cache statistics
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        
        return {
            "cache_size": len(self.token_cache),
            "max_cache_size": self.cache_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
        }
    
    def clear_cache(self):
        """
        Clear the tokenization cache.
        
        This can be useful for memory management or when switching between different tasks.
        """
        self.token_cache.clear()
        # Optionally reset statistics
        # self.cache_hits = 0
        # self.cache_misses = 0
    
    def resize_cache(self, new_size):
        """
        Resize the tokenization cache.
        
        Args:
            new_size: The new maximum size of the cache
            
        Returns:
            removed_count: Number of items removed from the cache
        """
        if new_size >= len(self.token_cache):
            # Cache doesn't need to be trimmed
            self.cache_size = new_size
            return 0
        
        # Calculate how many items to remove
        to_remove = len(self.token_cache) - new_size
        
        # Remove oldest items (in a more advanced implementation, this would use a proper LRU algorithm)
        for _ in range(to_remove):
            if self.token_cache:
                self.token_cache.pop(next(iter(self.token_cache)))
        
        # Update cache size
        self.cache_size = new_size
        
        return to_remove
