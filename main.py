"""
Main script for the recursive text editor.

This script demonstrates how to use the recursive text editor.
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoConfig

from rledit.models import BERTEditor, RecursiveEditor
from rledit.training import (
    EditEnvironment,
    LanguageModelPerplexity,
    GrammarCorrectness,
    CombinedQualityEvaluator,
)
from rledit.utils import setup_logger, log_example, log_edit_trace


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
    
    # Performance optimization arguments
    parser.add_argument(
        "--use_token_ids",
        action="store_true",
        help="Use token IDs directly instead of text for better performance",
    )
    parser.add_argument(
        "--cache_size",
        type=int,
        default=1000,
        help="Size of the tokenization cache (0 to disable)",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Stop processing examples that have already converged",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximum batch size to process at once (for memory constraints)",
    )
    parser.add_argument(
        "--use_tqdm",
        action="store_true",
        help="Show progress bar during editing",
    )
    
    # Input arguments
    parser.add_argument(
        "--input_text",
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
        "--log_file",
        type=str,
        default=None,
        help="Path to a log file",
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the edited text",
    )
    parser.add_argument(
        "--reference_file",
        type=str,
        default=None,
        help="Path to a file containing reference text for evaluation",
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


def create_recursive_editor(model, tokenizer, args):
    """Create the recursive editor."""
    return RecursiveEditor(
        editor_model=model,
        tokenizer=tokenizer,
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold,
        cache_size=args.cache_size,
    )


def create_quality_evaluator():
    """Create a quality evaluator."""
    evaluator = CombinedQualityEvaluator()
    
    # Add language model perplexity evaluator
    try:
        lm_perplexity = LanguageModelPerplexity()
        evaluator.add_evaluator("fluency", lm_perplexity, weight=0.5)
    except Exception as e:
        print(f"Warning: Failed to create language model perplexity evaluator: {e}")
    
    # Add grammar correctness evaluator
    try:
        grammar_correctness = GrammarCorrectness()
        evaluator.add_evaluator("grammar", grammar_correctness, weight=0.3)
    except Exception as e:
        print(f"Warning: Failed to create grammar correctness evaluator: {e}")
    
    return evaluator


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logger
    logger = setup_logger("rledit", args.log_file)
    logger.info("Recursive Text Editor")
    
    # Load the model
    logger.info(f"Loading model from {args.model_name_or_path}")
    model, tokenizer = load_model(args)
    
    # Create the recursive editor
    logger.info("Creating recursive editor")
    recursive_editor = create_recursive_editor(model, tokenizer, args)
    
    # Get the input text
    if args.input_text is not None:
        input_text = args.input_text
    elif args.input_file is not None:
        with open(args.input_file, "r") as f:
            input_text = f.read()
    else:
        input_text = input("Enter text to edit: ")
    
    # Edit the text
    logger.info("Editing text")
    if args.use_token_ids:
        logger.info("Using token IDs directly for better performance")
        # Tokenize the text
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True,
        ).to(args.device)
        
        # Edit the text using token IDs
        edited_text, edit_trace = recursive_editor.edit_until_convergence(
            inputs["input_ids"],
            sample=True,
            temperature=args.temperature,
            inputs_are_tokenized=True,
            attention_mask=inputs["attention_mask"],
            return_as_ids=False,
            early_stopping=args.early_stopping,
            max_batch_size=args.max_batch_size,
            use_tqdm=args.use_tqdm,
        )
    else:
        # Edit the text using the text interface
        edited_text, edit_trace = recursive_editor.edit_until_convergence(
            input_text,
            sample=True,
            temperature=args.temperature,
            early_stopping=args.early_stopping,
            max_batch_size=args.max_batch_size,
            use_tqdm=args.use_tqdm,
        )
    
    # Log the results
    log_example(logger, input_text, edited_text)
    log_edit_trace(logger, edit_trace)
    
    # Write the output
    if args.output_file is not None:
        with open(args.output_file, "w") as f:
            f.write(edited_text)
        logger.info(f"Edited text written to {args.output_file}")
    
    # Evaluate the edited text
    if args.evaluate:
        logger.info("Evaluating edited text")
        
        # Create a quality evaluator
        quality_evaluator = create_quality_evaluator()
        
        # Evaluate the original text
        original_quality = quality_evaluator(input_text)
        logger.info(f"Original text quality: {original_quality}")
        
        # Evaluate the edited text
        edited_quality = quality_evaluator(edited_text)
        logger.info(f"Edited text quality: {edited_quality}")
        
        # Compute the quality improvement
        quality_improvement = edited_quality - original_quality
        logger.info(f"Quality improvement: {quality_improvement}")
        
        # Evaluate against a reference if provided
        if args.reference_file is not None:
            with open(args.reference_file, "r") as f:
                reference_text = f.read()
            
            from rledit.utils import EvaluationMetrics
            
            # Compute BLEU score
            bleu_score = EvaluationMetrics.bleu_score(reference_text, edited_text)
            logger.info(f"BLEU score: {bleu_score}")
            
            # Compute edit distance
            edit_distance = EvaluationMetrics.edit_distance(reference_text, edited_text)
            logger.info(f"Edit distance: {edit_distance}")
            
            # Compute normalized edit distance
            normalized_edit_distance = EvaluationMetrics.normalized_edit_distance(
                reference_text, edited_text
            )
            logger.info(f"Normalized edit distance: {normalized_edit_distance}")
            
            # Check for exact match
            exact_match = EvaluationMetrics.exact_match(reference_text, edited_text)
            logger.info(f"Exact match: {exact_match}")
    
    # Print the results
    print("\nOriginal text:")
    print(input_text)
    print("\nEdited text:")
    print(edited_text)
    print("\nEdit trace:")
    for i, iteration_trace in enumerate(edit_trace):
        print(f"Iteration {i + 1}:")
        for j, op in enumerate(iteration_trace):
            print(f"  Operation {j + 1}: {op}")


if __name__ == "__main__":
    main()
