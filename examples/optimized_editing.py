"""
Optimized editing example for the recursive text editor.

This script demonstrates how to use the performance optimizations in the recursive text editor.
"""

import os
import time
import argparse
import torch
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm

from rledit.models import BERTEditor, RecursiveEditor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Optimized Recursive Text Editor")
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-uncased",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a checkpoint to load",
    )
    
    # Editor arguments
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=5,
        help="Maximum number of iterations",
    )
    parser.add_argument(
        "--convergence_threshold",
        type=float,
        default=0.95,
        help="Threshold for convergence (fraction of KEEP operations)",
    )
    
    # Benchmark arguments
    parser.add_argument(
        "--num_examples",
        type=int,
        default=100,
        help="Number of examples to process",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--text_length",
        type=int,
        default=100,
        help="Length of text examples",
    )
    
    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on",
    )
    
    return parser.parse_args()


def load_model(args):
    """Load the editor model."""
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    if args.checkpoint_path is not None:
        # Load the model from a checkpoint
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        config = checkpoint["config"]
        
        # Create the model
        model = BERTEditor.from_pretrained(args.model_name_or_path)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Load the model from pretrained
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        model = BERTEditor.from_pretrained(args.model_name_or_path)
    
    # Move the model to the device
    model.to(args.device)
    
    return model, tokenizer


def generate_example_texts(num_examples, text_length, tokenizer):
    """Generate example texts for benchmarking."""
    # Get a list of common words
    vocab_ids = list(range(1000, 5000))  # Skip special tokens
    
    # Generate random texts
    texts = []
    for _ in range(num_examples):
        # Generate random token IDs
        token_ids = torch.randint(
            low=min(vocab_ids),
            high=max(vocab_ids),
            size=(text_length,),
        ).tolist()
        
        # Convert to text
        text = tokenizer.decode(token_ids)
        texts.append(text)
    
    return texts


def benchmark_baseline(editor, texts, batch_size, device, tokenizer):
    """Benchmark the baseline implementation."""
    start_time = time.time()
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Edit the texts
        edited_texts, _ = editor.edit_until_convergence(
            batch_texts,
            sample=False,
        )
    
    end_time = time.time()
    return end_time - start_time


def benchmark_token_ids(editor, texts, batch_size, device, tokenizer):
    """Benchmark the token ID optimization."""
    start_time = time.time()
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize the texts
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True,
        ).to(device)
        
        # Edit the texts using token IDs
        edited_ids, _ = editor.edit_until_convergence(
            inputs["input_ids"],
            sample=False,
            inputs_are_tokenized=True,
            attention_mask=inputs["attention_mask"],
            return_as_ids=True,
        )
    
    end_time = time.time()
    return end_time - start_time


def benchmark_early_stopping(editor, texts, batch_size, device, tokenizer):
    """Benchmark the early stopping optimization."""
    start_time = time.time()
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Edit the texts with early stopping
        edited_texts, _ = editor.edit_until_convergence(
            batch_texts,
            sample=False,
            early_stopping=True,
        )
    
    end_time = time.time()
    return end_time - start_time


def benchmark_max_batch_size(editor, texts, device, tokenizer):
    """Benchmark the max batch size optimization."""
    start_time = time.time()
    
    # Edit all texts at once with max_batch_size
    edited_texts, _ = editor.edit_until_convergence(
        texts,
        sample=False,
        max_batch_size=16,
    )
    
    end_time = time.time()
    return end_time - start_time


def benchmark_combined(editor, texts, device, tokenizer):
    """Benchmark all optimizations combined."""
    start_time = time.time()
    
    # Tokenize all texts
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True,
    ).to(device)
    
    # Edit the texts with all optimizations
    edited_ids, _ = editor.edit_until_convergence(
        inputs["input_ids"],
        sample=False,
        inputs_are_tokenized=True,
        attention_mask=inputs["attention_mask"],
        return_as_ids=True,
        early_stopping=True,
        max_batch_size=16,
        use_tqdm=True,
    )
    
    end_time = time.time()
    return end_time - start_time


def benchmark_cache_sizes(editor, texts, device, tokenizer, cache_sizes):
    """Benchmark different cache sizes."""
    results = {}
    
    for cache_size in cache_sizes:
        # Set the cache size
        editor.resize_cache(cache_size)
        
        # Clear the cache
        editor.clear_cache()
        
        # Benchmark with this cache size
        start_time = time.time()
        
        # Tokenize all texts
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True,
        ).to(device)
        
        # Edit the texts
        edited_ids, _ = editor.edit_until_convergence(
            inputs["input_ids"],
            sample=False,
            inputs_are_tokenized=True,
            attention_mask=inputs["attention_mask"],
            return_as_ids=True,
            early_stopping=True,
            max_batch_size=16,
        )
        
        end_time = time.time()
        
        # Get cache statistics
        stats = editor.get_cache_stats()
        
        results[cache_size] = {
            "time": end_time - start_time,
            "hit_rate": stats["hit_rate"],
            "num_hits": stats["num_hits"],
            "num_misses": stats["num_misses"],
        }
    
    return results


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    print("Optimized Recursive Text Editor Benchmark")
    print("=========================================")
    
    # Load the model
    print(f"Loading model from {args.model_name_or_path}")
    model, tokenizer = load_model(args)
    
    # Create the recursive editor
    print("Creating recursive editor")
    editor = RecursiveEditor(
        editor_model=model,
        tokenizer=tokenizer,
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold,
        cache_size=1000,  # Default cache size
    )
    
    # Generate example texts
    print(f"Generating {args.num_examples} example texts of length {args.text_length}")
    texts = generate_example_texts(args.num_examples, args.text_length, tokenizer)
    
    # Benchmark baseline
    print("\nBenchmarking baseline implementation...")
    baseline_time = benchmark_baseline(editor, texts, args.batch_size, args.device, tokenizer)
    print(f"Baseline time: {baseline_time:.2f} seconds")
    
    # Benchmark token ID optimization
    print("\nBenchmarking token ID optimization...")
    token_id_time = benchmark_token_ids(editor, texts, args.batch_size, args.device, tokenizer)
    print(f"Token ID time: {token_id_time:.2f} seconds")
    print(f"Speedup: {baseline_time / token_id_time:.2f}x")
    
    # Benchmark early stopping
    print("\nBenchmarking early stopping optimization...")
    early_stopping_time = benchmark_early_stopping(editor, texts, args.batch_size, args.device, tokenizer)
    print(f"Early stopping time: {early_stopping_time:.2f} seconds")
    print(f"Speedup: {baseline_time / early_stopping_time:.2f}x")
    
    # Benchmark max batch size
    print("\nBenchmarking max batch size optimization...")
    max_batch_size_time = benchmark_max_batch_size(editor, texts, args.device, tokenizer)
    print(f"Max batch size time: {max_batch_size_time:.2f} seconds")
    print(f"Speedup: {baseline_time / max_batch_size_time:.2f}x")
    
    # Benchmark all optimizations combined
    print("\nBenchmarking all optimizations combined...")
    combined_time = benchmark_combined(editor, texts, args.device, tokenizer)
    print(f"Combined time: {combined_time:.2f} seconds")
    print(f"Speedup: {baseline_time / combined_time:.2f}x")
    
    # Benchmark different cache sizes
    print("\nBenchmarking different cache sizes...")
    cache_sizes = [0, 100, 500, 1000, 5000]
    cache_results = benchmark_cache_sizes(editor, texts, args.device, tokenizer, cache_sizes)
    
    print("\nCache size benchmark results:")
    print("----------------------------")
    print(f"{'Cache Size':<10} {'Time (s)':<10} {'Hit Rate':<10} {'Hits':<10} {'Misses':<10}")
    print("-" * 50)
    for cache_size, result in cache_results.items():
        print(f"{cache_size:<10} {result['time']:<10.2f} {result['hit_rate']:<10.2f} {result['num_hits']:<10} {result['num_misses']:<10}")
    
    # Find the optimal cache size
    optimal_cache_size = min(cache_results.items(), key=lambda x: x[1]["time"])[0]
    print(f"\nOptimal cache size: {optimal_cache_size}")
    
    # Print summary
    print("\nPerformance Optimization Summary:")
    print("-------------------------------")
    print(f"Baseline time:        {baseline_time:.2f} seconds")
    print(f"Token ID time:        {token_id_time:.2f} seconds ({baseline_time / token_id_time:.2f}x speedup)")
    print(f"Early stopping time:  {early_stopping_time:.2f} seconds ({baseline_time / early_stopping_time:.2f}x speedup)")
    print(f"Max batch size time:  {max_batch_size_time:.2f} seconds ({baseline_time / max_batch_size_time:.2f}x speedup)")
    print(f"Combined time:        {combined_time:.2f} seconds ({baseline_time / combined_time:.2f}x speedup)")
    
    print("\nRecommended configuration:")
    print("-------------------------")
    print(f"cache_size={optimal_cache_size}, early_stopping=True, max_batch_size=16, use_tqdm=True")


if __name__ == "__main__":
    main()
