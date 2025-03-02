"""
BERT-based editor model for the recursive text editor.

This module implements the BERT-based editor model that predicts edit operations
for input text.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel, AutoConfig

from .edit_operations import EditOperationHead, EditOperation


class BERTEditor(PreTrainedModel):
    """
    BERT-based editor model.
    
    This model uses a BERT-like encoder to predict token-level edit operations.
    It inherits from PreTrainedModel to leverage the HuggingFace ecosystem.
    """
    
    def __init__(self, config):
        """
        Initialize the BERT editor model.
        
        Args:
            config: The model configuration
        """
        super().__init__(config)
        
        # Load the encoder model
        self.encoder = AutoModel.from_config(config)
        
        # Create the edit operation head
        self.edit_head = EditOperationHead(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size
        )
        
        # Initialize weights
        self.init_weights()
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load a pretrained BERT model and add the edit operation head.
        
        Args:
            pretrained_model_name_or_path: The name or path of the pretrained model
            *model_args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            The initialized BERTEditor model
        """
        # Load the config
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs
        )
        
        # Create the model
        model = cls(config)
        
        # Load the encoder weights
        model.encoder = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs
        )
        
        return model
    
    def _prepare_encoder_inputs(self, input_ids, attention_mask, token_type_ids, position_ids,
                          head_mask, inputs_embeds, output_attentions, output_hidden_states,
                          return_dict):
        # Create base parameters that work with all versions
        params = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'inputs_embeds': inputs_embeds,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'return_dict': return_dict
        }
        
        # Only include token_type_ids, head_mask if explicitly provided and model supports it
        if token_type_ids is not None:
            try:
                # Try including token_type_ids
                params['token_type_ids'] = token_type_ids
                params['head_mask'] = head_mask
                # Verify model accepts this parameter
                self.encoder.forward(**params)
                return params
            except TypeError:
                # If TypeError occurs, model doesn't support token_type_ids
                pass
        
        return params
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass of the BERT editor model.
        
        Args:
            input_ids: The input token IDs, shape (batch_size, seq_len)
            attention_mask: The attention mask, shape (batch_size, seq_len)
            token_type_ids: The token type IDs, shape (batch_size, seq_len)
            position_ids: The position IDs, shape (batch_size, seq_len)
            head_mask: The head mask, shape (num_heads,) or (num_layers, num_heads)
            inputs_embeds: The input embeddings, shape (batch_size, seq_len, hidden_size)
            labels: The labels for computing the loss, shape (batch_size, seq_len)
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary
            
        Returns:
            A dictionary containing the model outputs
        """
        # Run the encoder
        encoder_outputs = self.encoder(**self._prepare_encoder_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ))
        
        # Get the hidden states
        hidden_states = encoder_outputs[0]
        
        # Predict edit operations
        operation_logits, replacement_logits, split_logits = self.edit_head(hidden_states)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Extract the operation labels and token labels
            operation_labels = labels[:, :, 0]
            replacement_labels = labels[:, :, 1]
            split_labels = labels[:, :, 2:4]
            
            # Compute operation loss
            operation_loss = nn.CrossEntropyLoss()(
                operation_logits.view(-1, len(EditOperation)),
                operation_labels.view(-1)
            )
            
            # Compute replacement loss for tokens with REPLACE operation
            replace_mask = (operation_labels == EditOperation.REPLACE.value).float()
            if replace_mask.sum() > 0:
                replacement_loss = nn.CrossEntropyLoss(reduction='none')(
                    replacement_logits.view(-1, self.config.vocab_size),
                    replacement_labels.view(-1)
                )
                replacement_loss = (replacement_loss * replace_mask.view(-1)).sum() / replace_mask.sum()
            else:
                replacement_loss = torch.tensor(0.0, device=operation_loss.device)
            
            # Compute split loss for tokens with SPLIT operation
            split_mask = (operation_labels == EditOperation.SPLIT.value).float()
            if split_mask.sum() > 0:
                # Reshape split logits to (batch_size, seq_len, 2, vocab_size)
                batch_size, seq_len, _ = split_logits.shape
                split_logits = split_logits.view(batch_size, seq_len, 2, -1)
                
                # Compute loss for first split token
                split_loss_1 = nn.CrossEntropyLoss(reduction='none')(
                    split_logits[:, :, 0].contiguous().view(-1, self.config.vocab_size),
                    split_labels[:, :, 0].contiguous().view(-1)
                )
                
                # Compute loss for second split token
                split_loss_2 = nn.CrossEntropyLoss(reduction='none')(
                    split_logits[:, :, 1].contiguous().view(-1, self.config.vocab_size),
                    split_labels[:, :, 1].contiguous().view(-1)
                )
                
                # Combine split losses
                split_loss = ((split_loss_1 + split_loss_2) * split_mask.view(-1)).sum() / split_mask.sum()
            else:
                split_loss = torch.tensor(0.0, device=operation_loss.device)
            
            # Combine losses
            loss = operation_loss + replacement_loss + split_loss
        
        # Return outputs
        return {
            "loss": loss,
            "operation_logits": operation_logits,
            "replacement_logits": replacement_logits,
            "split_logits": split_logits,
            "hidden_states": hidden_states,
            "encoder_outputs": encoder_outputs,
        }
    
    def predict_edit_operations(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        temperature=1.0,
        sample=False,
    ):
        """
        Predict edit operations for the input text.
        
        Args:
            input_ids: The input token IDs, shape (batch_size, seq_len)
            attention_mask: The attention mask, shape (batch_size, seq_len)
            token_type_ids: The token type IDs, shape (batch_size, seq_len)
            temperature: Temperature for sampling, higher values make the distribution more uniform
            sample: Whether to sample operations or take the argmax
            
        Returns:
            operations: The predicted operations, shape (batch_size, seq_len)
            replacements: The predicted replacement tokens, shape (batch_size, seq_len)
            splits: The predicted split tokens, shape (batch_size, seq_len, 2)
            operation_probs: The probabilities of the predicted operations, shape (batch_size, seq_len)
        """
        # Run the encoder
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            #token_type_ids=token_type_ids,
        )
        
        # Get the hidden states
        hidden_states = encoder_outputs[0]
        
        if sample:
            # Sample edit operations
            return self.edit_head.sample_operations(hidden_states, temperature)
        else:
            # Predict edit operations
            operation_logits, replacement_logits, split_logits = self.edit_head(hidden_states)
            
            # Get the most likely operations
            operations = torch.argmax(operation_logits, dim=-1)
            
            # Get the most likely replacements
            replacements = torch.argmax(replacement_logits, dim=-1)
            
            # Reshape split logits to (batch_size, seq_len, 2, vocab_size)
            batch_size, seq_len, _ = split_logits.shape
            split_logits = split_logits.view(batch_size, seq_len, 2, -1)
            
            # Get the most likely splits
            splits = torch.argmax(split_logits, dim=-1)
            
            # Get operation probabilities
            operation_probs = torch.softmax(operation_logits, dim=-1)
            operation_probs = torch.gather(
                operation_probs, -1, operations.unsqueeze(-1)
            ).squeeze(-1)
            
            # Get replacement probabilities
            replacement_probs = torch.softmax(replacement_logits, dim=-1)
            replacement_probs = torch.gather(
                replacement_probs, -1, replacements.unsqueeze(-1)
            ).squeeze(-1)
            
            # Get split probabilities
            split_probs = torch.softmax(split_logits, dim=-1)
            split_probs = torch.gather(
                split_probs, -1, splits.unsqueeze(-1)
            ).squeeze(-1)
            
            return operations, replacements, splits, operation_probs, replacement_probs, split_probs
