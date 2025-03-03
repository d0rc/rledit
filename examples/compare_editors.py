"""
Compare the performance of the original and optimized recursive editors.

This script demonstrates the performance improvements of the optimized recursive editor
compared to the original implementation.
"""

import os
import time
import argparse
import torch
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm

from rledit.models import BERTEditor, RecursiveEditor, RecursiveEditorOptimized


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare Recursive Editors")
    
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
        default=10,
        help="Number of examples to process",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--text_length",
        type=int,
        default=100,
        help="Length of text examples",
    )
    parser.add_argument(
        "--cache_size",
        type=int,
        default=1000,
        help="Size of the tokenization cache",
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


def benchmark_original_editor(editor, texts, batch_size, device, tokenizer):
    """Benchmark the original recursive editor."""
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


def benchmark_optimized_editor(editor, texts, batch_size, device, tokenizer):
    """Benchmark the optimized recursive editor."""
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


def benchmark_optimized_editor_with_features(editor, texts, batch_size, device, tokenizer):
    """Benchmark the optimized recursive editor with all optimization features enabled."""
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
        
        # Edit the texts with all optimizations
        edited_texts, _ = editor.edit_until_convergence(
            inputs["input_ids"],
            sample=False,
            inputs_are_tokenized=True,
            attention_mask=inputs["attention_mask"],
            early_stopping=True,
            max_batch_size=batch_size,
            use_tqdm=False,
        )
    
    end_time = time.time()
    return end_time - start_time


def compare_editors(args):
    """Compare the performance of the original and optimized recursive editors."""
    # Load the model
    print(f"Loading model from {args.model_name_or_path}")
    model, tokenizer = load_model(args)
    
    # Create the original recursive editor
    print("Creating original recursive editor")
    original_editor = RecursiveEditor(
        editor_model=model,
        tokenizer=tokenizer,
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold,
    )
    
    # Create the optimized recursive editor
    print("Creating optimized recursive editor")
    optimized_editor = RecursiveEditorOptimized(
        editor_model=model,
        tokenizer=tokenizer,
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold,
        cache_size=args.cache_size,
    )
    
    # Generate example texts
    print(f"Generating {args.num_examples} example texts of length {args.text_length}")
    texts = generate_example_texts(args.num_examples, args.text_length, tokenizer)
    
    # Benchmark original editor
    print("\nBenchmarking original recursive editor...")
    original_time = benchmark_original_editor(original_editor, texts, args.batch_size, args.device, tokenizer)
    print(f"Original editor time: {original_time:.2f} seconds")
    
    # Benchmark optimized editor (basic usage)
    print("\nBenchmarking optimized recursive editor (basic usage)...")
    optimized_time = benchmark_optimized_editor(optimized_editor, texts, args.batch_size, args.device, tokenizer)
    print(f"Optimized editor time: {optimized_time:.2f} seconds")
    print(f"Speedup: {original_time / optimized_time:.2f}x")
    
    # Benchmark optimized editor (with all features)
    print("\nBenchmarking optimized recursive editor (with all features)...")
    optimized_features_time = benchmark_optimized_editor_with_features(optimized_editor, texts, args.batch_size, args.device, tokenizer)
    print(f"Optimized editor (with features) time: {optimized_features_time:.2f} seconds")
    print(f"Speedup: {original_time / optimized_features_time:.2f}x")
    
    # Get cache statistics
    cache_stats = optimized_editor.get_cache_stats()
    print("\nCache statistics:")
    print(f"Cache size: {cache_stats['cache_size']}")
    print(f"Cache hits: {cache_stats['cache_hits']}")
    print(f"Cache misses: {cache_stats['cache_misses']}")
    print(f"Hit rate: {cache_stats['hit_rate']:.2f}")
    
    # Print summary
    print("\nPerformance Comparison Summary:")
    print("-------------------------------")
    print(f"Original editor time:                {original_time:.2f} seconds")
    print(f"Optimized editor time (basic):       {optimized_time:.2f} seconds ({original_time / optimized_time:.2f}x speedup)")
    print(f"Optimized editor time (with features): {optimized_features_time:.2f} seconds ({original_time / optimized_features_time:.2f}x speedup)")


def main():
    """Main function."""
    args = parse_args()
    
    print("Recursive Editor Performance Comparison")
    print("======================================")
    
    compare_editors(args)


if __name__ == "__main__":
    main()
