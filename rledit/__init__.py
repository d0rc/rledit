"""
Recursive BERT-based Text Editor with RL Training Pipeline.

This package implements a text editing model that recursively applies edits to input text
until convergence or maximum iterations. The model uses a BERT-like encoder to predict
token-level edit operations: KEEP, REMOVE, SPLIT, REPLACE.
"""

__version__ = "0.1.0"
