"""
RL trainer module for the recursive text editor.

This module implements the reinforcement learning training loop.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from ..models.recursive_editor import RecursiveEditor
from .environment import EditEnvironment


class RLTrainer:
    """
    Reinforcement learning trainer.
    
    This class handles the reinforcement learning training loop.
    """
    
    def __init__(
        self,
        editor_model,
        tokenizer,
        environment,
        config=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the RL trainer.
        
        Args:
            editor_model: The editor model
            tokenizer: The tokenizer
            environment: The edit environment
            config: The training configuration
            device: The device to run the training on
        """
        self.editor_model = editor_model
        self.tokenizer = tokenizer
        self.environment = environment
        self.config = config or {}
        self.device = device
        
        # Move the model to the device
        self.editor_model.to(self.device)
        
        # Create the recursive editor
        self.recursive_editor = RecursiveEditor(
            editor_model=self.editor_model,
            tokenizer=self.tokenizer,
            max_iterations=self.config.get("max_iterations", 5),
            convergence_threshold=self.config.get("convergence_threshold", 0.95),
            cache_size=self.config.get("cache_size", 1000),
        )
        
        # Create the optimizer
        self.optimizer = optim.AdamW(
            self.editor_model.parameters(),
            lr=self.config.get("learning_rate", 1e-5),
            weight_decay=self.config.get("weight_decay", 0.01),
        )
        
        # Create the learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get("rl_steps", 50000),
            eta_min=self.config.get("min_learning_rate", 1e-6),
        )
        
        # Create the tensorboard writer
        self.writer = SummaryWriter(self.config.get("log_dir", "logs"))
        
        # Initialize training state
        self.global_step = 0
        self.best_reward = float("-inf")
        self.best_model_path = None
    
    def train(self, train_dataloader, eval_dataloader=None, num_epochs=1, use_token_ids=True):
        """
        Train the model using reinforcement learning.
        
        Args:
            train_dataloader: The training data loader
            eval_dataloader: The evaluation data loader
            num_epochs: The number of epochs to train for
            use_token_ids: Whether to use token IDs directly instead of text
            
        Returns:
            The trained model
        """
        # Set the model to training mode
        self.editor_model.train()
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_reward = 0.0
            num_batches = 0
            
            # Create tqdm progress bar before the loop
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            
            # Process each batch
            for batch in progress_bar:
                # Extract the batch data
                original_texts = batch["original_texts"]
                
                # Move the batch to the device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids", None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                
                if use_token_ids:
                    # Run the recursive editor with action sampling using token IDs directly
                    edited_ids, edit_traces = self.recursive_editor.edit_with_sampling(
                        input_ids,
                        temperature=self.config.get("temperature", 1.0),
                        inputs_are_tokenized=True,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        return_as_ids=True,
                        early_stopping=self.config.get("early_stopping", True),
                        max_batch_size=self.config.get("max_batch_size", None),
                        use_tqdm=self.config.get("use_tqdm", False),
                    )
                    
                    # Convert edited IDs to text for reward computation
                    edited_texts = self.tokenizer.batch_decode(edited_ids, skip_special_tokens=True)
                else:
                    # Run the recursive editor with action sampling using text
                    edited_texts, edit_traces = self.recursive_editor.edit_with_sampling(
                        original_texts,
                        temperature=self.config.get("temperature", 1.0),
                        early_stopping=self.config.get("early_stopping", True),
                        max_batch_size=self.config.get("max_batch_size", None),
                        use_tqdm=self.config.get("use_tqdm", False),
                    )
                
                # Compute rewards
                rewards = self.environment.compute_batch_rewards(
                    original_texts, edited_texts, edit_traces
                )
                
                # Compute the RL loss
                rl_loss = self._compute_policy_gradient_loss(
                    edit_traces, rewards, discount_factor=self.config.get("discount_factor", 0.95)
                )
                
                # Update the model
                self.optimizer.zero_grad()
                rl_loss.backward()
                
                # Clip gradients
                if self.config.get("clip_grad_norm", None) is not None:
                    nn.utils.clip_grad_norm_(
                        self.editor_model.parameters(),
                        self.config.get("clip_grad_norm", 1.0),
                    )
                
                self.optimizer.step()
                self.scheduler.step()
                
                # Calculate batch reward
                batch_reward = sum(rewards) / len(rewards)
                
                # Update progress bar with current loss and reward
                progress_bar.set_postfix(loss=f"{rl_loss.item():.4f}", reward=f"{batch_reward:.4f}", refresh=True)
                
                # Update statistics
                epoch_loss += rl_loss.item()
                epoch_reward += batch_reward
                num_batches += 1
                self.global_step += 1
                
                # Log to tensorboard
                self.writer.add_scalar("Loss/train", rl_loss.item(), self.global_step)
                self.writer.add_scalar("Reward/train", sum(rewards) / len(rewards), self.global_step)
                self.writer.add_scalar("LearningRate", self.scheduler.get_last_lr()[0], self.global_step)
                
                # Evaluate periodically
                if eval_dataloader is not None and self.global_step % self.config.get("eval_steps", 1000) == 0:
                    eval_reward = self.evaluate(eval_dataloader, use_token_ids=use_token_ids)
                    self.writer.add_scalar("Reward/eval", eval_reward, self.global_step)
                    
                    # Save the best model
                    if eval_reward > self.best_reward:
                        self.best_reward = eval_reward
                        self.best_model_path = os.path.join(
                            self.config.get("output_dir", "output"),
                            f"best_model_step_{self.global_step}.pt",
                        )
                        self._save_model(self.best_model_path)
                
                # Save checkpoint periodically
                if self.global_step % self.config.get("save_steps", 10000) == 0:
                    checkpoint_path = os.path.join(
                        self.config.get("output_dir", "output"),
                        f"checkpoint_step_{self.global_step}.pt",
                    )
                    self._save_model(checkpoint_path)
            
            # Compute epoch statistics
            epoch_loss /= num_batches
            epoch_reward /= num_batches
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch statistics
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f} - Reward: {epoch_reward:.4f} - Time: {epoch_time:.2f}s")
            
            # Evaluate at the end of each epoch
            if eval_dataloader is not None:
                eval_reward = self.evaluate(eval_dataloader, use_token_ids=use_token_ids)
                print(f"Evaluation Reward: {eval_reward:.4f}")
                self.writer.add_scalar("Reward/eval", eval_reward, self.global_step)
                
                # Save the best model
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    self.best_model_path = os.path.join(
                        self.config.get("output_dir", "output"),
                        f"best_model_epoch_{epoch + 1}.pt",
                    )
                    self._save_model(self.best_model_path)
        
        # Save the final model
        final_model_path = os.path.join(
            self.config.get("output_dir", "output"),
            "final_model.pt",
        )
        self._save_model(final_model_path)
        
        # Close the tensorboard writer
        self.writer.close()
        
        return self.editor_model
    
    def evaluate(self, eval_dataloader, use_token_ids=True):
        """
        Evaluate the model.
        
        Args:
            eval_dataloader: The evaluation data loader
            use_token_ids: Whether to use token IDs directly instead of text
            
        Returns:
            The average reward
        """
        # Set the model to evaluation mode
        self.editor_model.eval()
        
        total_reward = 0.0
        num_examples = 0
        
        # Process each batch
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # Extract the batch data
                original_texts = batch["original_texts"]
                
                # Move the batch to the device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids", None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                
                if use_token_ids:
                    # Run the recursive editor using token IDs directly
                    edited_ids, edit_traces = self.recursive_editor.edit_until_convergence(
                        input_ids,
                        sample=False,
                        inputs_are_tokenized=True,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        return_as_ids=True,
                        early_stopping=self.config.get("early_stopping", True),
                        max_batch_size=self.config.get("max_batch_size", None),
                        use_tqdm=self.config.get("use_tqdm", False),
                    )
                    
                    # Convert edited IDs to text for reward computation
                    edited_texts = self.tokenizer.batch_decode(edited_ids, skip_special_tokens=True)
                else:
                    # Run the recursive editor using text
                    edited_texts, edit_traces = self.recursive_editor.edit_until_convergence(
                        original_texts,
                        sample=False,
                        early_stopping=self.config.get("early_stopping", True),
                        max_batch_size=self.config.get("max_batch_size", None),
                        use_tqdm=self.config.get("use_tqdm", False),
                    )
                
                # Compute rewards
                rewards = self.environment.compute_batch_rewards(
                    original_texts, edited_texts, edit_traces
                )
                
                # Update statistics
                total_reward += sum(rewards)
                num_examples += len(rewards)
        
        # Set the model back to training mode
        self.editor_model.train()
        
        # Compute the average reward
        avg_reward = total_reward / num_examples
        
        return avg_reward
    
    def _compute_policy_gradient_loss(self, edit_traces, rewards, discount_factor=0.95):
        """
        Compute the policy gradient loss.
        
        Args:
            edit_traces: List of edit traces
            rewards: List of rewards
            discount_factor: Discount factor for future rewards
            
        Returns:
            The policy gradient loss
        """
        # Pre-allocate lists to collect all probabilities and their associated rewards
        all_probs = []
        all_rewards = []
        
        # Precompute discount factors for efficiency (avoid repeated calculations)
        max_iterations = max((len(trace) for trace in edit_traces), default=0)
        discount_factors = torch.tensor(
            [discount_factor ** i for i in range(max_iterations)],
            device=self.device
        )
        
        # Process each example
        for trace_idx, (trace, reward) in enumerate(zip(edit_traces, rewards)):
            # Process each iteration
            for i, iteration_trace in enumerate(trace):
                # Get the precomputed discount factor
                discounted_reward = reward * discount_factors[i].item()
                
                # Collect all operation probabilities and their rewards
                for op in iteration_trace:
                    # Add the operation probability
                    all_probs.append(op["probability"])
                    all_rewards.append(discounted_reward)
                    
                    # Add the replacement probability if applicable
                    if op["operation"] == "REPLACE" and "replacement_probability" in op:
                        all_probs.append(op["replacement_probability"])
                        all_rewards.append(discounted_reward)
                    
                    # Add the split probabilities if applicable
                    if op["operation"] == "SPLIT" and "split_probabilities" in op:
                        all_probs.extend(op["split_probabilities"])
                        all_rewards.extend([discounted_reward] * len(op["split_probabilities"]))
        
        # If no probabilities were collected, return zero loss
        if not all_probs:
            return torch.zeros(1, device=self.device, requires_grad=True)
        
        # Convert collected data to tensors ONCE (not in a loop)
        prob_tensor = torch.tensor(all_probs, device=self.device, requires_grad=True)
        reward_tensor = torch.tensor(all_rewards, device=self.device)
        
        # Compute loss in a vectorized way
        log_probs = torch.log(prob_tensor)
        loss = -torch.sum(log_probs * reward_tensor) / len(rewards)
        
        return loss
    
    def _save_model(self, path):
        """
        Save the model.
        
        Args:
            path: The path to save the model to
        """
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        torch.save(
            {
                "model_state_dict": self.editor_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "best_reward": self.best_reward,
                "config": self.config,
            },
            path,
        )
        
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load the model.
        
        Args:
            path: The path to load the model from
            
        Returns:
            The loaded model
        """
        # Load the checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load the model state
        self.editor_model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load the optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load the scheduler state
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load the training state
        self.global_step = checkpoint["global_step"]
        self.best_reward = checkpoint["best_reward"]
        self.config = checkpoint["config"]
        
        print(f"Model loaded from {path}")
        
        return self.editor_model
