"""
Simple example of using the recursive text editor.

This script demonstrates how to use the recursive text editor with a pretrained model.
"""

import argparse
import time
import torch
from transformers import AutoTokenizer

from rledit.models import BERTEditor, RecursiveEditor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simple example of using the recursive text editor")
    parser.add_argument(
        "--use_token_ids",
        action="store_true",
        help="Use token IDs directly instead of text for better performance",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a benchmark to compare performance with and without token ID optimization",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations for the benchmark",
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Load the model and tokenizer
    model_name = "bert-base-uncased"
    print(f"Loading model from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BERTEditor.from_pretrained(model_name)
    
    # Create the recursive editor
    print("Creating recursive editor")
    editor = RecursiveEditor(
        editor_model=model,
        tokenizer=tokenizer,
        max_iterations=5,
        convergence_threshold=0.95,
    )
    
    # Example texts with errors
    texts = [
        "This is a example of text with some errors.",
        "The cat sat on the mat and it was happy.",
        "I have went to the store yesterday and buyed some milk.",
        "She dont like to eat vegetables but she like fruits.",
    ]
    
    if args.benchmark:
        run_benchmark(editor, texts, args.iterations)
    else:
        # Edit each text
        for i, text in enumerate(texts):
            print(f"\nExample {i + 1}:")
            print(f"Original: {text}")
            
            if args.use_token_ids:
                # Tokenize the text
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    return_attention_mask=True,
                )
                
                # Edit the text using token IDs
                edited_ids, edit_trace = editor.edit_until_convergence(
                    inputs["input_ids"],
                    inputs_are_tokenized=True,
                    attention_mask=inputs["attention_mask"],
                    return_as_ids=False,
                )
                print(f"Edited: {edited_ids}")
            else:
                # Edit the text using the text interface
                edited_text, edit_trace = editor.edit_until_convergence(text)
                print(f"Edited: {edited_text}")
            
            # Print the edit trace
            print("Edit trace:")
            for j, iteration_trace in enumerate(edit_trace):
                print(f"  Iteration {j + 1}:")
                for k, op in enumerate(iteration_trace):
                    print(f"    Operation {k + 1}: {op}")


def run_benchmark(editor, texts, iterations):
    """Run a benchmark to compare performance with and without token ID optimization."""
    print(f"\nRunning benchmark with {iterations} iterations...")
    
    # Tokenize texts once for token ID optimization
    tokenizer = editor.tokenizer
    tokenized_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True,
    )
    
    # Benchmark without token ID optimization
    start_time = time.time()
    for _ in range(iterations):
        for text in texts:
            editor.edit_until_convergence(text)
    text_time = time.time() - start_time
    print(f"Time without token ID optimization: {text_time:.4f} seconds")
    
    # Benchmark with token ID optimization
    start_time = time.time()
    for _ in range(iterations):
        for i in range(len(texts)):
            editor.edit_until_convergence(
                tokenized_inputs["input_ids"][i:i+1],
                inputs_are_tokenized=True,
                attention_mask=tokenized_inputs["attention_mask"][i:i+1],
                return_as_ids=True,
            )
    token_id_time = time.time() - start_time
    print(f"Time with token ID optimization: {token_id_time:.4f} seconds")
    
    # Calculate speedup
    speedup = text_time / token_id_time
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
