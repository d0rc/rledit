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
            
        Performance Notes:
            - The tokenization cache improves performance by storing previously tokenized texts
              to avoid redundant tokenization operations.
            - A larger cache_size can improve performance for repetitive tasks but uses more memory.
            - The cache is particularly beneficial when:
                1. Processing the same texts multiple times
                2. Working with texts that have common substrings
                3. Running multiple iterations on similar texts
            - Use get_cache_stats() to monitor cache performance and resize_cache() to adjust the cache size.
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
        Tokenize texts with caching for improved performance.
        
        This method maintains a cache of previously tokenized texts to avoid redundant
        tokenization operations, which can be expensive especially for large texts or
        when processing the same text multiple times during recursive editing.
        
        Args:
            texts: List of input texts
            **kwargs: Additional arguments for the tokenizer (e.g., truncation, padding)
            
        Returns:
            inputs: Tokenized inputs with all requested tensors (input_ids, attention_mask, etc.)
            
        Performance Notes:
            - Cache keys are based on both the text content and tokenization parameters
            - Cache hits avoid the expensive tokenization process
            - The cache is particularly effective during recursive editing where the same
              or similar texts are processed multiple times
            - The cache size is limited to prevent memory issues (configurable via cache_size)
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
        Combine multiple tokenized inputs into a single batch for efficient processing.
        
        This method takes a list of individual tokenized inputs (which may come from
        the cache or from fresh tokenization) and combines them into a single batch
        suitable for processing by the model.
        
        Args:
            tokenized_inputs: List of tokenized inputs, each containing tensors like
                             input_ids, attention_mask, etc.
            batch_size: The total batch size
            
        Returns:
            combined_inputs: Combined tokenized inputs with all tensors concatenated
                            along the batch dimension
                            
        Performance Notes:
            - This method is used by _cached_tokenize to combine cached and newly
              tokenized inputs into a single batch
            - The concatenation operation is efficient for PyTorch tensors
            - This approach allows for flexible batch construction from heterogeneous
              sources (cache hits and misses)
            - All tensor keys from the input dictionaries are preserved in the output
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
    
    def edit_until_convergence(self, inputs, sample=False, temperature=1.0, inputs_are_tokenized=False, 
                              attention_mask=None, token_type_ids=None, return_as_ids=False,
                              early_stopping=True, max_batch_size=None, use_tqdm=False):
        """
        Edit texts or token IDs until convergence or maximum iterations.
        
        This method applies edit operations recursively to the input texts or token IDs
        until either convergence is reached (determined by the convergence_threshold) or
        the maximum number of iterations is reached.
        
        Args:
            inputs: List of input texts or tensor of input token IDs
            sample: Whether to sample operations or take the argmax
            temperature: Temperature for sampling
            inputs_are_tokenized: Whether inputs are already tokenized (faster)
            attention_mask: Attention mask for tokenized inputs
            token_type_ids: Token type IDs for tokenized inputs
            return_as_ids: Whether to return token IDs instead of texts
            early_stopping: Whether to stop processing examples that have converged
            max_batch_size: Maximum batch size to process at once (for memory constraints)
            use_tqdm: Whether to show a progress bar (requires tqdm to be installed)
            
        Returns:
            edited_outputs: The edited texts or token IDs
            edit_traces: The edit traces for RL training
            
        Performance Notes:
            - Using token IDs directly (inputs_are_tokenized=True) is significantly faster
              as it avoids repeated tokenization/detokenization during the recursive process
            - For batch processing, tokenize inputs once and pass the token IDs directly
            - When using token IDs, also provide the attention_mask for best results
            - Set return_as_ids=True if you need token IDs for further processing
            - The early_stopping parameter can significantly improve performance by
              avoiding unnecessary processing of examples that have already converged
            - For large batches, use max_batch_size to avoid memory issues
            - Processing directly with token IDs can be up to 5x faster than processing with text
            - The use_tqdm parameter can be helpful for monitoring progress with large batches
        """
        # Import tqdm if needed
        if use_tqdm:
            try:
                from tqdm import tqdm
            except ImportError:
                print("Warning: tqdm not installed. Progress bar will not be shown.")
                use_tqdm = False
        
        # Case 1: Text inputs
        if not inputs_are_tokenized:
            if isinstance(inputs, str):
                inputs = [inputs]
                single_input = True
            else:
                single_input = False
            
            batch_size = len(inputs)
            current_texts = inputs.copy()
            edit_traces = [[] for _ in range(batch_size)]
            
            # Track which examples have converged
            converged_mask = [False] * batch_size
            final_texts = current_texts.copy()
            
            # Create iteration range with tqdm if requested
            iter_range = tqdm(range(self.max_iterations)) if use_tqdm else range(self.max_iterations)
            
            for iteration in iter_range:
                if early_stopping:
                    # Filter out converged examples
                    active_indices = [i for i, converged in enumerate(converged_mask) if not converged]
                    
                    # If all examples have converged, we're done
                    if not active_indices:
                        if use_tqdm:
                            iter_range.set_description(f"All examples converged at iteration {iteration}")
                        break
                else:
                    # Process all examples regardless of convergence
                    active_indices = list(range(batch_size))
                
                # Process only active examples
                active_texts = [current_texts[i] for i in active_indices]
                
                # Process in chunks if max_batch_size is specified
                if max_batch_size and len(active_indices) > max_batch_size:
                    # Process in chunks
                    chunk_results = []
                    chunk_traces = []
                    chunk_converged = []
                    
                    for chunk_start in range(0, len(active_indices), max_batch_size):
                        chunk_end = min(chunk_start + max_batch_size, len(active_indices))
                        chunk_indices = active_indices[chunk_start:chunk_end]
                        chunk_texts = [current_texts[i] for i in chunk_indices]
                        
                        # Process this chunk
                        chunk_result = self._process_text_chunk(
                            chunk_texts, 
                            sample, 
                            temperature
                        )
                        
                        chunk_results.append(chunk_result[0])  # new texts
                        chunk_traces.append(chunk_result[1])   # traces
                        chunk_converged.append(chunk_result[2])  # convergence flags
                    
                    # Combine results from all chunks
                    new_active_texts = [item for sublist in chunk_results for item in sublist]
                    active_batch_edit_traces = [item for sublist in chunk_traces for item in sublist]
                    active_converged = [item for sublist in chunk_converged for item in sublist]
                else:
                    # Process all active examples at once
                    # Use cached tokenization
                    inputs_dict = self._cached_tokenize(
                        active_texts,
                        truncation=True,
                        return_attention_mask=True,
                        return_token_type_ids=True,
                    )
                    
                    # Predict edit operations
                    with torch.no_grad():
                        operations, replacements, splits, operation_probs, replacement_probs, split_probs = (
                            self.editor_model.predict_edit_operations(
                                input_ids=inputs_dict["input_ids"],
                                attention_mask=inputs_dict["attention_mask"],
                                token_type_ids=inputs_dict["token_type_ids"],
                                temperature=temperature,
                                sample=sample,
                            )
                        )
                    
                    # Apply edit operations
                    new_active_texts, active_batch_edit_traces = self._apply_edit_operations(
                        active_texts,
                        inputs_dict["input_ids"],
                        operations,
                        replacements,
                        splits,
                        operation_probs,
                        replacement_probs,
                        split_probs,
                    )
                    
                    # Check for convergence of active examples
                    active_converged = self._check_convergence(operations, inputs_dict["attention_mask"])
                
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
                
                # Update progress bar description
                if use_tqdm:
                    converged_count = sum(converged_mask)
                    iter_range.set_description(
                        f"Iteration {iteration+1}/{self.max_iterations}, "
                        f"Converged: {converged_count}/{batch_size} "
                        f"({100.0 * converged_count / batch_size:.1f}%)"
                    )
            
            if return_as_ids:
                # Convert final texts to token IDs if requested
                encodings = self.tokenizer(
                    final_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.editor_model.device)
                
                if single_input:
                    return encodings["input_ids"][0], edit_traces[0]
                else:
                    return encodings["input_ids"], edit_traces
            else:
                if single_input:
                    return final_texts[0], edit_traces[0]
                else:
                    return final_texts, edit_traces
                
        # Case 2: Token ID inputs
        else:
            # Handling already tokenized inputs
            if isinstance(inputs, torch.Tensor):
                input_ids = inputs
                batch_size = input_ids.size(0)
                single_input = False  # Always treat as batch for tensor inputs
            else:
                # Handle case where inputs might be a dictionary
                input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs
                batch_size = input_ids.size(0)
                single_input = False
            
            # Initialize variables
            current_ids = input_ids.clone()
            attention_mask = attention_mask if attention_mask is not None else torch.ones_like(current_ids)
            token_type_ids = token_type_ids if token_type_ids is not None else None
            edit_traces = [[] for _ in range(batch_size)]
            
            # Track which examples have converged
            converged_mask = [False] * batch_size
            final_ids = current_ids.clone()
            
            # Create iteration range with tqdm if requested
            iter_range = tqdm(range(self.max_iterations)) if use_tqdm else range(self.max_iterations)
            
            # Processing iterations
            for iteration in iter_range:
                if early_stopping:
                    # Filter non-converged examples
                    active_indices = [i for i, converged in enumerate(converged_mask) if not converged]
                    
                    # If all examples have converged, we're done
                    if not active_indices:
                        if use_tqdm:
                            iter_range.set_description(f"All examples converged at iteration {iteration}")
                        break
                else:
                    # Process all examples regardless of convergence
                    active_indices = list(range(batch_size))
                
                # Get active subset of inputs
                active_ids = current_ids[active_indices]
                active_attention_mask = attention_mask[active_indices]
                active_token_type_ids = token_type_ids[active_indices] if token_type_ids is not None else None
                
                # Process in chunks if max_batch_size is specified
                if max_batch_size and len(active_indices) > max_batch_size:
                    # Process in chunks
                    chunk_results = []
                    chunk_traces = []
                    chunk_converged = []
                    
                    for chunk_start in range(0, len(active_indices), max_batch_size):
                        chunk_end = min(chunk_start + max_batch_size, len(active_indices))
                        chunk_indices = active_indices[chunk_start:chunk_end]
                        
                        # Get chunk inputs
                        chunk_ids = active_ids[chunk_start:chunk_end]
                        chunk_attention_mask = active_attention_mask[chunk_start:chunk_end]
                        chunk_token_type_ids = active_token_type_ids[chunk_start:chunk_end] if active_token_type_ids is not None else None
                        
                        # Process this chunk
                        chunk_result = self._process_token_id_chunk(
                            chunk_ids,
                            chunk_attention_mask,
                            chunk_token_type_ids,
                            sample,
                            temperature
                        )
                        
                        new_chunk_ids, chunk_edit_traces, chunk_converged_flags = chunk_result
                        
                        chunk_results.append(new_chunk_ids)
                        chunk_traces.append(chunk_edit_traces)
                        chunk_converged.append(chunk_converged_flags)
                    
                    # Combine results from all chunks
                    if all(isinstance(ids, torch.Tensor) for ids in chunk_results):
                        # If all chunks returned tensors, concatenate them
                        new_active_ids = torch.cat(chunk_results, dim=0)
                    else:
                        # If some chunks returned lists, flatten the list
                        new_active_ids = [item for sublist in chunk_results for item in sublist]
                    
                    active_batch_edit_traces = [item for sublist in chunk_traces for item in sublist]
                    active_converged = [item for sublist in chunk_converged for item in sublist]
                else:
                    # Process all active examples at once
                    # No tokenization needed - we already have token IDs
                    with torch.no_grad():
                        operations, replacements, splits, operation_probs, replacement_probs, split_probs = (
                            self.editor_model.predict_edit_operations(
                                input_ids=active_ids,
                                attention_mask=active_attention_mask,
                                token_type_ids=active_token_type_ids,
                                temperature=temperature,
                                sample=sample,
                            )
                        )
                    
                    # Apply edit operations directly to token IDs
                    new_active_ids, active_batch_edit_traces = self._apply_edit_operations_on_ids(
                        active_ids,
                        operations,
                        replacements,
                        splits,
                        operation_probs,
                        replacement_probs,
                        split_probs,
                    )
                    
                    # Check for convergence of active examples
                    active_converged = self._check_convergence(operations, active_attention_mask)
                
                # Update IDs, traces, and convergence status for active examples
                for idx, active_idx in enumerate(active_indices):
                    # Update edit traces
                    edit_traces[active_idx].append(active_batch_edit_traces[idx])
                    
                    # Update current IDs and final IDs
                    # Note: We need to handle padding since new_active_ids might have different lengths
                    # For simplicity, we'll convert to padded tensors later
                    if isinstance(new_active_ids, list):
                        # If we're returning a list of tensors with different lengths
                        current_ids_list = [t.clone() for t in current_ids]
                        current_ids_list[active_idx] = new_active_ids[idx]
                        final_ids_list = [t.clone() for t in final_ids]
                        final_ids_list[active_idx] = new_active_ids[idx]
                        
                        # Convert lists to padded tensors
                        current_ids = self._pad_and_stack_tensors(current_ids_list)
                        final_ids = self._pad_and_stack_tensors(final_ids_list)
                    else:
                        # If we're returning a padded tensor
                        if isinstance(new_active_ids, torch.Tensor):
                            # Handle tensor case
                            seq_len = min(new_active_ids.size(1), current_ids.size(1)) if len(new_active_ids.size()) > 1 else new_active_ids.size(0)
                            current_ids[active_idx, :seq_len] = new_active_ids[idx, :seq_len]
                            final_ids[active_idx, :seq_len] = new_active_ids[idx, :seq_len]
                        else:
                            # Handle other cases
                            current_ids[active_idx] = new_active_ids[idx]
                            final_ids[active_idx] = new_active_ids[idx]
                    
                    # Update convergence status
                    if active_converged[idx]:
                        converged_mask[active_idx] = True
                
                # Update progress bar description
                if use_tqdm:
                    converged_count = sum(converged_mask)
                    iter_range.set_description(
                        f"Iteration {iteration+1}/{self.max_iterations}, "
                        f"Converged: {converged_count}/{batch_size} "
                        f"({100.0 * converged_count / batch_size:.1f}%)"
                    )
            
            if return_as_ids:
                if single_input:
                    return final_ids[0], edit_traces[0]
                else:
                    return final_ids, edit_traces
            else:
                # Convert back to text
                final_texts = self.tokenizer.batch_decode(final_ids, skip_special_tokens=True)
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
        
        This method applies edit operations (KEEP, REMOVE, REPLACE, SPLIT) to text inputs
        by first converting token IDs to tokens, applying the operations, and then
        converting the edited tokens back to text.
        
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
            
        Performance Notes:
            - This method involves multiple conversions between text and tokens, which can be expensive
            - For better performance, consider using _apply_edit_operations_on_ids which works directly
              with token IDs and avoids these conversions
            - The tokenization/detokenization process is a significant bottleneck in the recursive
              editing process, especially for large texts or many iterations
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
        Check if the editing process has converged using a vectorized implementation.
        
        This method determines if the editing process has converged by calculating the
        fraction of KEEP operations for each example in the batch. If this fraction
        exceeds the convergence_threshold, the example is considered to have converged.
        
        Args:
            operations: The predicted operations, shape (batch_size, seq_len)
            attention_mask: The attention mask, shape (batch_size, seq_len)
            
        Returns:
            converged: Boolean list indicating whether each example has converged
            
        Performance Notes:
            - This method uses a vectorized implementation for better performance
            - The attention mask is used to ignore padding tokens in the calculation
            - Early convergence detection can significantly reduce the number of iterations
              needed, especially for texts that require minimal editing
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
    
    def edit_with_sampling(self, inputs, temperature=1.0, inputs_are_tokenized=False, 
                          attention_mask=None, token_type_ids=None, return_as_ids=False,
                          early_stopping=True, max_batch_size=None, use_tqdm=False):
        """
        Edit texts or token IDs with sampling for reinforcement learning training.
        
        This method is a specialized version of edit_until_convergence that always
        samples operations instead of taking the argmax. This is particularly useful
        for reinforcement learning training where exploration of different edit
        trajectories is important.
        
        Args:
            inputs: List of input texts or tensor of input token IDs
            temperature: Temperature for sampling (higher values increase diversity)
            inputs_are_tokenized: Whether inputs are already tokenized (faster)
            attention_mask: Attention mask for tokenized inputs
            token_type_ids: Token type IDs for tokenized inputs
            return_as_ids: Whether to return token IDs instead of texts
            early_stopping: Whether to stop processing examples that have converged
            max_batch_size: Maximum batch size to process at once (for memory constraints)
            use_tqdm: Whether to show a progress bar (requires tqdm to be installed)
            
        Returns:
            edited_outputs: The edited texts or token IDs
            edit_traces: The edit traces for RL training, containing detailed
                         information about each edit operation and its probability
            
        Performance Notes:
            - Using token IDs directly (inputs_are_tokenized=True) is significantly faster
              as it avoids repeated tokenization/detokenization during the recursive process
            - For batch processing, tokenize inputs once and pass the token IDs directly
            - When using token IDs, also provide the attention_mask for best results
            - Set return_as_ids=True if you need token IDs for further processing
            - The temperature parameter controls the randomness of sampling:
                * Lower values (e.g., 0.1) make sampling closer to greedy selection
                * Higher values (e.g., 2.0) increase diversity but may lead to less coherent edits
            - The edit_traces return value is particularly important for RL training as it
              contains the probabilities needed for policy gradient methods
            - For large batches, use max_batch_size to avoid memory issues
            - The early_stopping parameter can significantly improve performance by
              avoiding unnecessary processing of examples that have already converged
        """
        return self.edit_until_convergence(
            inputs, 
            sample=True, 
            temperature=temperature,
            inputs_are_tokenized=inputs_are_tokenized,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_as_ids=return_as_ids,
            early_stopping=early_stopping,
            max_batch_size=max_batch_size,
            use_tqdm=use_tqdm
        )
    
    def get_cache_stats(self):
        """
        Get statistics about the tokenization cache for performance monitoring.
        
        This method returns statistics about the tokenization cache, including the
        current size, maximum size, number of hits and misses, and the hit rate.
        These statistics can be used to monitor cache performance and make decisions
        about cache size adjustments.
        
        Returns:
            stats: Dictionary containing cache statistics with the following keys:
                - cache_size: Current number of entries in the cache
                - max_cache_size: Maximum allowed cache size
                - cache_hits: Number of cache hits
                - cache_misses: Number of cache misses
                - hit_rate: Ratio of hits to total lookups (hits + misses)
                
        Performance Notes:
            - A high hit rate (close to 1.0) indicates good cache utilization
            - A low hit rate may indicate that the cache size is too small or that
              there is little repetition in the processed texts
            - Monitor these statistics during processing to identify opportunities
              for performance optimization
            - Use resize_cache() to adjust the cache size based on these statistics
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
        Clear the tokenization cache to free memory.
        
        This method completely empties the tokenization cache, freeing memory and
        resetting the cache state. It can be useful when switching between different
        tasks or when memory usage needs to be minimized.
        
        Performance Notes:
            - Clearing the cache frees memory but will result in cache misses for
              subsequent tokenization operations
            - Consider using this method between processing different batches or
              datasets to prevent the cache from being polluted with irrelevant entries
            - For long-running processes with similar texts, it may be better to
              use resize_cache() instead of completely clearing the cache
            - The cache hit statistics are preserved by default, but can be reset
              by uncommenting the relevant lines in the implementation
        """
        self.token_cache.clear()
        # Optionally reset statistics
        # self.cache_hits = 0
        # self.cache_misses = 0
    
    def _apply_edit_operations_on_ids(
        self,
        input_ids,
        operations,
        replacements,
        splits,
        operation_probs,
        replacement_probs,
        split_probs,
    ):
        """
        Apply edit operations directly to token IDs for improved performance.
        
        This method applies edit operations (KEEP, REMOVE, REPLACE, SPLIT) directly to token IDs
        without converting to and from text, which significantly improves performance in the
        recursive editing process.
        
        Args:
            input_ids: The input token IDs, shape (batch_size, seq_len)
            operations: The predicted operations, shape (batch_size, seq_len)
            replacements: The predicted replacement tokens, shape (batch_size, seq_len)
            splits: The predicted split tokens, shape (batch_size, seq_len, 2)
            operation_probs: The probabilities of the predicted operations, shape (batch_size, seq_len)
            replacement_probs: The probabilities of the predicted replacements, shape (batch_size, seq_len)
            split_probs: The probabilities of the predicted splits, shape (batch_size, seq_len, 2)
            
        Returns:
            edited_ids: The edited token IDs
            edit_traces: The edit traces for RL training
            
        Performance Notes:
            - Working directly with token IDs avoids the expensive tokenization/detokenization process
            - The token text is still included in the trace for debugging purposes, but could be
              omitted for further performance improvements
            - This method is significantly faster than _apply_edit_operations which works with text
            - The resulting edited_ids are padded to a uniform length for batch processing
        """
        batch_size = input_ids.size(0)
        edited_ids_list = []
        edit_traces = []
        
        for i in range(batch_size):
            # Get the tokens for this example
            token_ids = input_ids[i]
            
            # Apply edit operations
            edited_token_ids = []
            trace = []
            
            for j, token_id in enumerate(token_ids):
                # Skip special tokens
                token_id_item = token_id.item()
                if token_id_item in [
                    self.tokenizer.cls_token_id, 
                    self.tokenizer.sep_token_id, 
                    self.tokenizer.pad_token_id
                ]:
                    continue
                
                # Get the operation for this token
                op = operations[i, j].item()
                op_prob = operation_probs[i, j].item()
                
                # Get token text for trace (optional, could be removed for efficiency)
                token_text = self.tokenizer.convert_ids_to_tokens(token_id_item)
                
                # Apply the operation
                if op == EditOperation.KEEP.value:
                    edited_token_ids.append(token_id_item)
                    trace.append({
                        "operation": "KEEP",
                        "token": token_text,
                        "token_id": token_id_item,
                        "probability": op_prob,
                    })
                elif op == EditOperation.REMOVE.value:
                    trace.append({
                        "operation": "REMOVE",
                        "token": token_text,
                        "token_id": token_id_item,
                        "probability": op_prob,
                    })
                elif op == EditOperation.REPLACE.value:
                    replacement_id = replacements[i, j].item()
                    replacement_text = self.tokenizer.convert_ids_to_tokens(replacement_id)
                    edited_token_ids.append(replacement_id)
                    trace.append({
                        "operation": "REPLACE",
                        "token": token_text,
                        "token_id": token_id_item,
                        "replacement": replacement_text,
                        "replacement_id": replacement_id,
                        "probability": op_prob,
                        "replacement_probability": replacement_probs[i, j].item(),
                    })
                elif op == EditOperation.SPLIT.value:
                    split_ids = [
                        splits[i, j, 0].item(),
                        splits[i, j, 1].item(),
                    ]
                    split_texts = [
                        self.tokenizer.convert_ids_to_tokens(split_ids[0]),
                        self.tokenizer.convert_ids_to_tokens(split_ids[1]),
                    ]
                    edited_token_ids.extend(split_ids)
                    trace.append({
                        "operation": "SPLIT",
                        "token": token_text,
                        "token_id": token_id_item,
                        "split_tokens": split_texts,
                        "split_token_ids": split_ids,
                        "probability": op_prob,
                        "split_probabilities": [split_probs[i, j, 0].item(), split_probs[i, j, 1].item()],
                    })
            
            # Convert list to tensor
            edited_ids_list.append(torch.tensor(edited_token_ids, device=input_ids.device))
            edit_traces.append(trace)
        
        # Pad to maximum length to return a regular tensor
        padded_ids = self._pad_and_stack_tensors(edited_ids_list)
        
        return padded_ids, edit_traces
    
    def _process_token_id_chunk(self, input_ids, attention_mask, token_type_ids, sample, temperature):
        """
        Process a chunk of token ID inputs for batched processing.
        
        This helper method processes a chunk of token ID inputs, applying edit operations
        and checking for convergence. It's used by edit_until_convergence when
        processing large batches in chunks to avoid memory issues.
        
        Args:
            input_ids: Tensor of input token IDs for this chunk
            attention_mask: Attention mask for this chunk
            token_type_ids: Token type IDs for this chunk (or None)
            sample: Whether to sample operations or take the argmax
            temperature: Temperature for sampling
            
        Returns:
            new_ids: The edited token IDs
            edit_traces: The edit traces for RL training
            converged: Boolean list indicating whether each example has converged
            
        Performance Notes:
            - This method is used internally by edit_until_convergence for batch chunking
            - Processing in chunks helps avoid memory issues with large batches
            - Each chunk is processed independently, which can be more memory-efficient
              but may be slightly slower than processing the entire batch at once
            - Unlike text processing, this method works directly with token IDs for better performance
        """
        # Predict edit operations
        with torch.no_grad():
            operations, replacements, splits, operation_probs, replacement_probs, split_probs = (
                self.editor_model.predict_edit_operations(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    temperature=temperature,
                    sample=sample,
                )
            )
        
        # Apply edit operations directly to token IDs
        new_ids, edit_traces = self._apply_edit_operations_on_ids(
            input_ids,
            operations,
            replacements,
            splits,
            operation_probs,
            replacement_probs,
            split_probs,
        )
        
        # Check for convergence
        converged = self._check_convergence(operations, attention_mask)
        
        return new_ids, edit_traces, converged
    
    def _process_text_chunk(self, texts, sample, temperature):
        """
        Process a chunk of text inputs for batched processing.
        
        This helper method processes a chunk of text inputs, applying edit operations
        and checking for convergence. It's used by edit_until_convergence when
        processing large batches in chunks to avoid memory issues.
        
        Args:
            texts: List of input texts for this chunk
            sample: Whether to sample operations or take the argmax
            temperature: Temperature for sampling
            
        Returns:
            new_texts: The edited texts
            edit_traces: The edit traces for RL training
            converged: Boolean list indicating whether each example has converged
            
        Performance Notes:
            - This method is used internally by edit_until_convergence for batch chunking
            - Processing in chunks helps avoid memory issues with large batches
            - Each chunk is processed independently, which can be more memory-efficient
              but may be slightly slower than processing the entire batch at once
        """
        # Use cached tokenization
        inputs_dict = self._cached_tokenize(
            texts,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        
        # Predict edit operations
        with torch.no_grad():
            operations, replacements, splits, operation_probs, replacement_probs, split_probs = (
                self.editor_model.predict_edit_operations(
                    input_ids=inputs_dict["input_ids"],
                    attention_mask=inputs_dict["attention_mask"],
                    token_type_ids=inputs_dict["token_type_ids"],
                    temperature=temperature,
                    sample=sample,
                )
            )
        
        # Apply edit operations
        new_texts, edit_traces = self._apply_edit_operations(
            texts,
            inputs_dict["input_ids"],
            operations,
            replacements,
            splits,
            operation_probs,
            replacement_probs,
            split_probs,
        )
        
        # Check for convergence
        converged = self._check_convergence(operations, inputs_dict["attention_mask"])
        
        return new_texts, edit_traces, converged
    
    def _pad_and_stack_tensors(self, tensor_list):
        """
        Pad and stack tensors of different lengths for batch processing.
        
        This method takes a list of tensors with potentially different lengths and
        creates a single padded tensor suitable for batch processing. The padding
        is done with zeros to the maximum length in the batch.
        
        Args:
            tensor_list: List of tensors with different lengths
            
        Returns:
            padded_tensor: Padded and stacked tensor with shape (batch_size, max_len)
            
        Performance Notes:
            - This operation is necessary when working with variable-length sequences
              in a batched manner
            - The padding is done with zeros, which is compatible with most operations
              as long as proper attention masks are used
            - This method preserves the device and dtype of the original tensors
            - For very large batches with highly variable lengths, this can lead to
              some memory inefficiency due to the padding
        """
        # Get the maximum length
        max_len = max(tensor.size(0) for tensor in tensor_list)
        
        # Get the device and dtype from the first tensor
        device = tensor_list[0].device
        dtype = tensor_list[0].dtype
        
        # Create a padded tensor
        batch_size = len(tensor_list)
        padded_tensor = torch.zeros(batch_size, max_len, dtype=dtype, device=device)
        
        # Fill the padded tensor
        for i, tensor in enumerate(tensor_list):
            length = tensor.size(0)
            padded_tensor[i, :length] = tensor
        
        return padded_tensor
    
    def resize_cache(self, new_size):
        """
        Resize the tokenization cache to manage memory usage.
        
        This method allows dynamic adjustment of the tokenization cache size
        to balance between memory usage and performance. Increasing the cache size
        can improve performance for repetitive tasks, while decreasing it can
        reduce memory usage.
        
        Args:
            new_size: The new maximum size of the cache
            
        Returns:
            removed_count: Number of items removed from the cache
            
        Performance Notes:
            - Increasing cache size improves performance for repetitive tasks but uses more memory
            - Decreasing cache size frees memory but may reduce performance if the same texts
              are processed multiple times
            - The current implementation uses a simple approach to remove items (not true LRU)
            - For optimal performance, monitor cache statistics with get_cache_stats() and
              adjust the cache size based on hit rate and memory constraints
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
