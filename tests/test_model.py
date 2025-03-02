"""
Test script for the recursive text editor.

This script tests the basic functionality of the recursive text editor.
"""

import unittest
import torch
from transformers import AutoTokenizer

from rledit.models import BERTEditor, RecursiveEditor, EditOperation
from rledit.utils import TokenizationUtils


class TestBERTEditor(unittest.TestCase):
    """Test the BERTEditor model."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        # Load the model and tokenizer
        cls.model_name = "bert-base-uncased"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.model = BERTEditor.from_pretrained(cls.model_name)
        
        # Create the recursive editor
        cls.editor = RecursiveEditor(
            editor_model=cls.model,
            tokenizer=cls.tokenizer,
            max_iterations=3,
            convergence_threshold=0.95,
        )
    
    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.model.encoder)
        self.assertIsNotNone(self.model.edit_head)
    
    def test_edit_operations(self):
        """Test the edit operations."""
        # Check that the edit operations are defined correctly
        self.assertEqual(EditOperation.KEEP.value, 0)
        self.assertEqual(EditOperation.REMOVE.value, 1)
        self.assertEqual(EditOperation.SPLIT.value, 2)
        self.assertEqual(EditOperation.REPLACE.value, 3)
    
    def test_forward_pass(self):
        """Test the forward pass of the model."""
        # Create a simple input
        text = "This is a test."
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Run the forward pass
        outputs = self.model(**inputs)
        
        # Check the outputs
        self.assertIn("operation_logits", outputs)
        self.assertIn("replacement_logits", outputs)
        self.assertIn("split_logits", outputs)
        self.assertIn("hidden_states", outputs)
        
        # Check the shapes
        batch_size, seq_len = inputs["input_ids"].shape
        self.assertEqual(outputs["operation_logits"].shape, (batch_size, seq_len, len(EditOperation)))
        self.assertEqual(outputs["replacement_logits"].shape, (batch_size, seq_len, self.tokenizer.vocab_size))
    
    def test_predict_edit_operations(self):
        """Test the prediction of edit operations."""
        # Create a simple input
        text = "This is a test."
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Predict edit operations
        operations, replacements, splits, operation_probs, replacement_probs, split_probs = (
            self.model.predict_edit_operations(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs.get("token_type_ids", None),
            )
        )
        
        # Check the shapes
        batch_size, seq_len = inputs["input_ids"].shape
        self.assertEqual(operations.shape, (batch_size, seq_len))
        self.assertEqual(replacements.shape, (batch_size, seq_len))
        self.assertEqual(splits.shape, (batch_size, seq_len, 2))
        self.assertEqual(operation_probs.shape, (batch_size, seq_len))
        self.assertEqual(replacement_probs.shape, (batch_size, seq_len))
        self.assertEqual(split_probs.shape, (batch_size, seq_len, 2))
    
    def test_recursive_editor(self):
        """Test the recursive editor."""
        # Create a simple input
        text = "This is a test."
        
        # Edit the text
        edited_text, edit_trace = self.editor.edit_until_convergence(text)
        
        # Check the results
        self.assertIsNotNone(edited_text)
        self.assertIsNotNone(edit_trace)
        self.assertIsInstance(edited_text, str)
        self.assertIsInstance(edit_trace, list)
    
    def test_tokenization_utils(self):
        """Test the tokenization utilities."""
        # Create a simple input
        text = "This is a test."
        
        # Create the tokenization utilities
        tokenization_utils = TokenizationUtils(self.tokenizer)
        
        # Test the token to character mappings
        token_to_char = tokenization_utils.get_token_to_char_mappings(text)
        self.assertIsNotNone(token_to_char)
        self.assertIsInstance(token_to_char, dict)
        
        # Test the character to token mappings
        char_to_token = tokenization_utils.get_char_to_token_mappings(text)
        self.assertIsNotNone(char_to_token)
        self.assertIsInstance(char_to_token, dict)
        
        # Test the word to token mappings
        word_to_tokens = tokenization_utils.get_word_to_token_mappings(text)
        self.assertIsNotNone(word_to_tokens)
        self.assertIsInstance(word_to_tokens, dict)
        
        # Test the token to word mappings
        token_to_word = tokenization_utils.get_token_to_word_mappings(text)
        self.assertIsNotNone(token_to_word)
        self.assertIsInstance(token_to_word, dict)
    
    def test_apply_token_level_edits(self):
        """Test applying token-level edits."""
        # Create a simple input
        text = "This is a test."
        
        # Create the tokenization utilities
        tokenization_utils = TokenizationUtils(self.tokenizer)
        
        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"][0]
        
        # Create some edit operations
        operations = torch.zeros_like(input_ids)  # All KEEP operations
        
        # Apply the edits
        edited_text = tokenization_utils.apply_token_level_edits(text, operations)
        
        # Check the results
        self.assertIsNotNone(edited_text)
        self.assertIsInstance(edited_text, str)


if __name__ == "__main__":
    unittest.main()
