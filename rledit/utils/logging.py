"""
Logging utilities for the recursive text editor.

This module provides utilities for logging.
"""

import os
import logging
import sys
from datetime import datetime


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Set up a logger.
    
    Args:
        name: The name of the logger
        log_file: The path to the log file (optional)
        level: The logging level
        
    Returns:
        logger: The logger
    """
    # Create the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create the formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Create the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create the file handler if a log file is specified
    if log_file is not None:
        # Create the log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create the file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_timestamp():
    """
    Get a timestamp string.
    
    Returns:
        timestamp: The timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log_config(logger, config):
    """
    Log a configuration.
    
    Args:
        logger: The logger
        config: The configuration
    """
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")


def log_metrics(logger, metrics, prefix=""):
    """
    Log metrics.
    
    Args:
        logger: The logger
        metrics: The metrics
        prefix: The prefix for the metrics
    """
    logger.info(f"{prefix} Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")


def log_example(logger, original_text, edited_text, reference_text=None):
    """
    Log an example.
    
    Args:
        logger: The logger
        original_text: The original text
        edited_text: The edited text
        reference_text: The reference text (optional)
    """
    logger.info("Example:")
    logger.info(f"  Original: {original_text}")
    logger.info(f"  Edited: {edited_text}")
    if reference_text is not None:
        logger.info(f"  Reference: {reference_text}")


def log_edit_trace(logger, edit_trace):
    """
    Log an edit trace.
    
    Args:
        logger: The logger
        edit_trace: The edit trace
    """
    logger.info("Edit Trace:")
    for i, iteration_trace in enumerate(edit_trace):
        logger.info(f"  Iteration {i + 1}:")
        for j, op in enumerate(iteration_trace):
            logger.info(f"    Operation {j + 1}: {op}")
