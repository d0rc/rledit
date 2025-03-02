"""
Supervised pretraining example for the recursive text editor.

This script demonstrates how to pretrain the recursive text editor using supervised learning
before fine-tuning with reinforcement learning.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
from tqdm import tqdm

from rledit.models import BERTEditor
from rledit.data import EditDataset, EditCollator
from rledit.utils import setup_logger, get_timestamp, log_config, log_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Supervised Pretraining for Recursive Text Editor")
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-uncased",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="jfleg",
        help="The name of the dataset to use",
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
        "--learning_rate",
        type=float,
        default=5e-5,
        help="The initial learning rate for AdamW",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Run evaluation every X steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X updates steps",
    )
    
    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
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
    
    # Load the model from pretrained
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = BERTEditor.from_pretrained(args.model_name_or_path)
    
    # Move the model to the device
    model.to(args.device)
    
    return model, tokenizer


def load_datasets(args, tokenizer):
    """Load the datasets."""
    # Load the dataset
    raw_datasets = load_dataset(args.dataset_name)
    
    # Preprocess the datasets
    if args.dataset_name == "jfleg":
        # JFLEG dataset has original and corrected texts
        train_dataset = raw_datasets["validation"] if "validation" in raw_datasets else raw_datasets["train"]
        eval_dataset = raw_datasets["test"] if "test" in raw_datasets else raw_datasets["validation"]
        
        # Extract the texts
        train_original_texts = [example["sentence"] for example in train_dataset]
        train_edited_texts = [example["corrections"][0] for example in train_dataset]
        eval_original_texts = [example["sentence"] for example in eval_dataset]
        eval_edited_texts = [example["corrections"][0] for example in eval_dataset]
    else:
        # For other datasets, we need to adapt accordingly
        raise ValueError(f"Dataset {args.dataset_name} not supported for supervised pretraining")
    
    # Limit the number of samples if specified
    if args.max_train_samples is not None:
        train_original_texts = train_original_texts[:args.max_train_samples]
        train_edited_texts = train_edited_texts[:args.max_train_samples]
    if args.max_val_samples is not None:
        eval_original_texts = eval_original_texts[:args.max_val_samples]
        eval_edited_texts = eval_edited_texts[:args.max_val_samples]
    
    # Create the edit datasets
    train_edit_dataset = EditDataset(
        original_texts=train_original_texts,
        edited_texts=train_edited_texts,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    eval_edit_dataset = EditDataset(
        original_texts=eval_original_texts,
        edited_texts=eval_edited_texts,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    return train_edit_dataset, eval_edit_dataset


def train(args, model, tokenizer, train_dataset, eval_dataset, logger):
    """Train the model."""
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
    
    # Create the optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Create the learning rate scheduler
    num_training_steps = len(train_dataloader) * args.num_train_epochs
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps,
        eta_min=1e-6,
    )
    
    # Training loop
    global_step = 0
    best_eval_loss = float("inf")
    
    for epoch in range(args.num_train_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_train_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        for batch in tqdm(train_dataloader, desc="Training"):
            # Move the batch to the device
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(args.device)
            edit_labels = batch["edit_labels"].to(args.device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=edit_labels,
            )
            
            loss = outputs["loss"]
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            # Update statistics
            train_loss += loss.item()
            train_steps += 1
            global_step += 1
            
            # Log training loss
            if global_step % args.logging_steps == 0:
                logger.info(f"Step {global_step} - Training loss: {loss.item():.4f}")
            
            # Evaluate
            if global_step % args.eval_steps == 0:
                eval_loss = evaluate(args, model, eval_dataloader)
                logger.info(f"Step {global_step} - Evaluation loss: {eval_loss:.4f}")
                
                # Save the best model
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_model_path = os.path.join(args.output_dir, f"best_model_step_{global_step}.pt")
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": lr_scheduler.state_dict(),
                            "global_step": global_step,
                            "epoch": epoch,
                            "config": vars(args),
                        },
                        best_model_path,
                    )
                    logger.info(f"Best model saved to {best_model_path}")
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint_step_{global_step}.pt")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": lr_scheduler.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                        "config": vars(args),
                    },
                    checkpoint_path,
                )
                logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Compute epoch statistics
        train_loss /= train_steps
        logger.info(f"Epoch {epoch + 1}/{args.num_train_epochs} - Training loss: {train_loss:.4f}")
        
        # Evaluate at the end of each epoch
        eval_loss = evaluate(args, model, eval_dataloader)
        logger.info(f"Epoch {epoch + 1}/{args.num_train_epochs} - Evaluation loss: {eval_loss:.4f}")
        
        # Save the best model
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_model_path = os.path.join(args.output_dir, f"best_model_epoch_{epoch + 1}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": lr_scheduler.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "config": vars(args),
                },
                best_model_path,
            )
            logger.info(f"Best model saved to {best_model_path}")
    
    # Save the final model
    final_model_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "global_step": global_step,
            "epoch": args.num_train_epochs - 1,
            "config": vars(args),
        },
        final_model_path,
    )
    logger.info(f"Final model saved to {final_model_path}")
    
    return model


def evaluate(args, model, eval_dataloader):
    """Evaluate the model."""
    model.eval()
    eval_loss = 0.0
    eval_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # Move the batch to the device
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(args.device)
            edit_labels = batch["edit_labels"].to(args.device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=edit_labels,
            )
            
            loss = outputs["loss"]
            
            # Update statistics
            eval_loss += loss.item()
            eval_steps += 1
    
    # Compute average loss
    eval_loss /= eval_steps
    
    return eval_loss


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logger
    timestamp = get_timestamp()
    log_file = os.path.join(args.log_dir, f"pretrain_{timestamp}.log")
    logger = setup_logger("rledit", log_file)
    logger.info("Supervised Pretraining for Recursive Text Editor")
    
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
    
    # Train the model
    logger.info("Training model")
    model = train(args, model, tokenizer, train_dataset, eval_dataset, logger)
    
    logger.info("Supervised pretraining completed")


if __name__ == "__main__":
    main()
