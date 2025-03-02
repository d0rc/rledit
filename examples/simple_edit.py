"""
Simple example of using the recursive text editor.

This script demonstrates how to use the recursive text editor with a pretrained model.
"""

import torch
from transformers import AutoTokenizer

from rledit.models import BERTEditor, RecursiveEditor


def main():
    """Main function."""
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
    
    # Edit each text
    for i, text in enumerate(texts):
        print(f"\nExample {i + 1}:")
        print(f"Original: {text}")
        
        # Edit the text
        edited_text, edit_trace = editor.edit_until_convergence(text)
        print(f"Edited: {edited_text}")
        
        # Print the edit trace
        print("Edit trace:")
        for j, iteration_trace in enumerate(edit_trace):
            print(f"  Iteration {j + 1}:")
            for k, op in enumerate(iteration_trace):
                print(f"    Operation {k + 1}: {op}")


if __name__ == "__main__":
    main()
