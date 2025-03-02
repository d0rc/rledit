"""
Training script for the recursive text editor.

This script demonstrates how to train the recursive text editor using reinforcement learning.
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset

from rledit.models import BERTEditor, RecursiveEditor
from rledit.data import EditDataset, EditCollator
from rledit.training import (
    RLTrainer,
    EditEnvironment,
    LanguageModelPerplexity,
    GrammarCorrectness,
    CombinedQualityEvaluator,
    LanguageModelPerplexityReward,
    GrammarCorrectnessReward,
    IterationEfficiencyReward,
    EditDistanceReward,
    CombinedReward,
)
from rledit.utils import setup_logger, get_timestamp, log_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Recursive Text Editor")
    
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
    
    # Data arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="The name of the dataset to use",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="wikitext-103-raw-v1",
        help="The configuration name of the dataset to use",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="Path to a file containing training data",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="Path to a file containing validation data",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Maximum number of training examples to use",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        help="Maximum number of validation examples to use",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions and checkpoints will be written",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="The directory where the logs will be written",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size per GPU/TPU core/CPU for training",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size per GPU/TPU core/CPU for evaluation",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="The initial learning rate for AdamW",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Linear warmup over warmup_steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Log every X updates steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="Run evaluation every X steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
        help="Save checkpoint every X updates steps",
    )
    
    # RL arguments
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
    parser.add_argument(
        "--discount_factor",
        type=float,
        default=0.95,
        help="Discount factor for future rewards",
    )
    parser.add_argument(
        "--efficiency_factor",
        type=float,
        default=0.1,
        help="Weight for the efficiency bonus",
    )
    
    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps",
        help="Device to run the model on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization",
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


def load_datasets(args, tokenizer):
    """Load the datasets."""
    # Load the dataset
    if args.train_file is not None and args.validation_file is not None:
        # Load from files
        data_files = {
            "train": args.train_file,
            "validation": args.validation_file,
        }
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    else:
        # Load from Hugging Face datasets
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    
    # Preprocess the datasets
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    
    # Create the datasets
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]
    
    # Limit the number of samples if specified
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    if args.max_val_samples is not None:
        eval_dataset = eval_dataset.select(range(args.max_val_samples))
    
    # Extract the texts
    train_texts = [example[text_column_name] for example in train_dataset]
    eval_texts = [example[text_column_name] for example in eval_dataset]
    
    # Create the edit datasets
    train_edit_dataset = EditDataset(
        original_texts=train_texts,
        edited_texts=train_texts,  # Use the same texts for now
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    eval_edit_dataset = EditDataset(
        original_texts=eval_texts,
        edited_texts=eval_texts,  # Use the same texts for now
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    return train_edit_dataset, eval_edit_dataset


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


def create_reward_function(args):
    """Create a reward function."""
    reward = CombinedReward()
    
    # Add language model perplexity reward
    try:
        lm_perplexity = LanguageModelPerplexity()
        lm_perplexity_reward = LanguageModelPerplexityReward(lm_perplexity)
        reward.add_reward_function("fluency", lm_perplexity_reward, weight=0.5)
    except Exception as e:
        print(f"Warning: Failed to create language model perplexity reward: {e}")
    
    # Add grammar correctness reward
    try:
        grammar_correctness = GrammarCorrectness()
        grammar_correctness_reward = GrammarCorrectnessReward(grammar_correctness)
        reward.add_reward_function("grammar", grammar_correctness_reward, weight=0.3)
    except Exception as e:
        print(f"Warning: Failed to create grammar correctness reward: {e}")
    
    # Add iteration efficiency reward
    iteration_efficiency_reward = IterationEfficiencyReward(
        max_iterations=args.max_iterations,
        efficiency_factor=args.efficiency_factor,
    )
    reward.add_reward_function("efficiency", iteration_efficiency_reward, weight=0.2)
    
    # Add edit distance reward
    edit_distance_reward = EditDistanceReward()
    reward.add_reward_function("edit_distance", edit_distance_reward, weight=0.1)
    
    return reward


def create_environment(args, reward_function=None):
    """Create the environment."""
    # Create a quality evaluator if no reward function is provided
    if reward_function is None:
        quality_evaluator = create_quality_evaluator()
    else:
        quality_evaluator = None
    
    # Create the environment
    environment = EditEnvironment(
        quality_evaluator=quality_evaluator,
        max_iterations=args.max_iterations,
        efficiency_factor=args.efficiency_factor,
        reward_weights={
            "fluency": 0.5,
            "grammar": 0.3,
            "efficiency": 0.2,
        },
    )
    
    return environment


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logger
    timestamp = get_timestamp()
    log_file = os.path.join(args.log_dir, f"train_{timestamp}.log")
    logger = setup_logger("rledit", log_file)
    logger.info("Training Recursive Text Editor")
    
    # Log the configuration
    log_config(logger, vars(args))
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model
    logger.info(f"Loading model from {args.model_name_or_path}")
    model, tokenizer = load_model(args)
    
    # Load the datasets
    logger.info("Loading datasets")
    train_dataset, eval_dataset = load_datasets(args, tokenizer)
    
    # Create the data loaders
    collator = EditCollator(tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    
    # Create the environment
    logger.info("Creating environment")
    reward_function = create_reward_function(args)
    environment = create_environment(args, reward_function)
    
    # Create the trainer
    logger.info("Creating trainer")
    trainer = RLTrainer(
        editor_model=model,
        tokenizer=tokenizer,
        environment=environment,
        config={
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "max_iterations": args.max_iterations,
            "convergence_threshold": args.convergence_threshold,
            "temperature": args.temperature,
            "discount_factor": args.discount_factor,
            "efficiency_factor": args.efficiency_factor,
            "log_dir": args.log_dir,
            "output_dir": args.output_dir,
            "logging_steps": args.logging_steps,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
        },
        device=args.device,
    )
    
    # Train the model
    logger.info("Training model")
    trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        num_epochs=args.num_train_epochs,
    )
    
    # Save the final model
    final_model_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
        },
        final_model_path,
    )
    logger.info(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
