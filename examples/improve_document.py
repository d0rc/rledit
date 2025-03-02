"""
Document improvement example for the recursive text editor.

This script demonstrates how to use the recursive text editor to improve the quality of a text document.
"""

import os
import argparse
import torch
from transformers import AutoTokenizer

from rledit.models import BERTEditor, RecursiveEditor
from rledit.training import (
    LanguageModelPerplexity,
    GrammarCorrectness,
    CombinedQualityEvaluator,
)
from rledit.utils import setup_logger, log_example, log_edit_trace


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Improve Document Quality")
    
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
        "--input_file",
        type=str,
        required=True,
        help="Path to the input document",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output document",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to the log file",
    )
    
    # Processing arguments
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Size of text chunks to process",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap between chunks",
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the quality improvement",
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
        
        # Create the model
        model = BERTEditor.from_pretrained(args.model_name_or_path)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Load the model from pretrained
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


def chunk_text(text, chunk_size, overlap):
    """
    Split text into chunks with overlap.
    
    Args:
        text: The text to split
        chunk_size: The size of each chunk
        overlap: The overlap between chunks
        
    Returns:
        chunks: List of text chunks
    """
    # Split the text into sentences
    sentences = text.split(". ")
    
    # Add periods back to sentences
    sentences = [sentence + "." if not sentence.endswith(".") else sentence for sentence in sentences]
    
    # Initialize chunks
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Process each sentence
    for sentence in sentences:
        # Get the length of the sentence
        sentence_length = len(sentence.split())
        
        # Check if adding this sentence would exceed the chunk size
        if current_length + sentence_length > chunk_size and current_chunk:
            # Add the current chunk to the list of chunks
            chunks.append(" ".join(current_chunk))
            
            # Start a new chunk with overlap
            overlap_size = min(overlap, len(current_chunk))
            current_chunk = current_chunk[-overlap_size:]
            current_length = sum(len(sentence.split()) for sentence in current_chunk)
        
        # Add the sentence to the current chunk
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def merge_chunks(chunks, overlap):
    """
    Merge chunks back into a single text.
    
    Args:
        chunks: List of text chunks
        overlap: The overlap between chunks
        
    Returns:
        text: The merged text
    """
    if not chunks:
        return ""
    
    # Initialize the merged text with the first chunk
    merged_text = chunks[0]
    
    # Process each chunk
    for i in range(1, len(chunks)):
        # Get the current chunk
        chunk = chunks[i]
        
        # Split the chunks into words
        merged_words = merged_text.split()
        chunk_words = chunk.split()
        
        # Find the overlap
        overlap_size = min(overlap, len(merged_words), len(chunk_words))
        
        # Find the best overlap point
        best_overlap = 0
        best_position = len(merged_words)
        
        for j in range(1, overlap_size + 1):
            # Check the overlap
            if merged_words[-j:] == chunk_words[:j]:
                # Update the best overlap
                best_overlap = j
                best_position = len(merged_words) - j
        
        # Merge the chunks
        if best_overlap > 0:
            merged_text = " ".join(merged_words[:best_position] + chunk_words)
        else:
            merged_text = merged_text + " " + chunk
    
    return merged_text


def improve_document(args, editor, logger, quality_evaluator=None):
    """
    Improve the quality of a document.
    
    Args:
        args: The command line arguments
        editor: The recursive editor
        logger: The logger
        quality_evaluator: The quality evaluator
        
    Returns:
        improved_text: The improved text
    """
    # Read the input document
    with open(args.input_file, "r") as f:
        text = f.read()
    
    # Log the original text
    logger.info(f"Original document length: {len(text)} characters")
    
    # Evaluate the original text
    if quality_evaluator is not None:
        original_quality = quality_evaluator(text)
        logger.info(f"Original text quality: {original_quality}")
    
    # Split the text into chunks
    chunks = chunk_text(text, args.chunk_size, args.overlap)
    logger.info(f"Split document into {len(chunks)} chunks")
    
    # Improve each chunk
    improved_chunks = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i + 1}/{len(chunks)}")
        
        # Improve the chunk
        improved_chunk, edit_trace = editor.edit_until_convergence(
            chunk,
            sample=True,
            temperature=args.temperature,
        )
        
        # Log the improvement
        log_example(logger, chunk, improved_chunk)
        log_edit_trace(logger, edit_trace)
        
        # Add the improved chunk
        improved_chunks.append(improved_chunk)
    
    # Merge the improved chunks
    improved_text = merge_chunks(improved_chunks, args.overlap)
    logger.info(f"Improved document length: {len(improved_text)} characters")
    
    # Evaluate the improved text
    if quality_evaluator is not None:
        improved_quality = quality_evaluator(improved_text)
        logger.info(f"Improved text quality: {improved_quality}")
        
        # Compute the quality improvement
        quality_improvement = improved_quality - original_quality
        logger.info(f"Quality improvement: {quality_improvement}")
    
    # Write the improved text
    with open(args.output_file, "w") as f:
        f.write(improved_text)
    logger.info(f"Improved document written to {args.output_file}")
    
    return improved_text


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logger
    logger = setup_logger("rledit", args.log_file)
    logger.info("Document Quality Improvement")
    
    # Load the model
    logger.info(f"Loading model from {args.model_name_or_path}")
    model, tokenizer = load_model(args)
    
    # Create the recursive editor
    logger.info("Creating recursive editor")
    editor = create_recursive_editor(model, tokenizer, args)
    
    # Create the quality evaluator
    quality_evaluator = None
    if args.evaluate:
        logger.info("Creating quality evaluator")
        quality_evaluator = create_quality_evaluator()
    
    # Improve the document
    logger.info("Improving document")
    improved_text = improve_document(args, editor, logger, quality_evaluator)
    
    logger.info("Document improvement completed")


if __name__ == "__main__":
    main()
