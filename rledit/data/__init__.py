"""
Data module for the recursive text editor.

This module contains the implementation of the data handling components,
including datasets and data loaders.
"""

from .dataset import EditDataset, EditCollator

__all__ = ["EditDataset", "EditCollator"]
