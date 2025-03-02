"""
Command-line interface for the recursive text editor.

This module provides a command-line interface for the recursive text editor.
"""

import argparse
import sys
import torch
from transformers import AutoTokenizer

from rledit.models import BERTEditor, RecursiveEditor
from rledit.utils import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Recursive Text Editor")
    
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling",
    )
    
    # Input arguments
    parser.add_argument(
        "input",
        nargs="?",
        type=str,
        default=None,
        help="Input text to edit",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to a file containing input text",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to a file to write the edited text",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    
    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on",
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logger
    logger = setup_logger("rledit", level="INFO" if args.verbose else "WARNING")
    
    # Load the model and tokenizer
    logger.info(f"Loading model from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    if args.checkpoint_path is not None:
        # Load the model from a checkpoint
        checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
        
        # Create the model
        model = BERTEditor.from_pretrained(args.model_name_or_path)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Load the model from pretrained
        model = BERTEditor.from_pretrained(args.model_name_or_path)
    
    # Move the model to the device
    model.to(args.device)
    
    # Create the recursive editor
    logger.info("Creating recursive editor")
    editor = RecursiveEditor(
        editor_model=model,
        tokenizer=tokenizer,
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold,
    )
    
    # Get the input text
    if args.input is not None:
        input_text = args.input
    elif args.input_file is not None:
        with open(args.input_file, "r") as f:
            input_text = f.read()
    else:
        # Read from stdin
        input_text = sys.stdin.read()
    
    # Edit the text
    logger.info("Editing text")
    edited_text, edit_trace = editor.edit_until_convergence(
        input_text,
        sample=True,
        temperature=args.temperature,
    )
    
    # Write the output
    if args.output_file is not None:
        with open(args.output_file, "w") as f:
            f.write(edited_text)
        logger.info(f"Edited text written to {args.output_file}")
    else:
        # Write to stdout
        print(edited_text)
    
    # Print the edit trace if verbose
    if args.verbose:
        print("\nEdit trace:")
        for i, iteration_trace in enumerate(edit_trace):
            print(f"Iteration {i + 1}:")
            for j, op in enumerate(iteration_trace):
                print(f"  Operation {j + 1}: {op}")


if __name__ == "__main__":
    main()
