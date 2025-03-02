# Recursive BERT-based Text Editor with RL Training Pipeline

This project implements a text editing model that recursively applies edits to input text until convergence or maximum iterations. The model uses a BERT-like encoder to predict token-level edit operations: KEEP, REMOVE, SPLIT, REPLACE. It is trained using reinforcement learning to maximize text quality improvements across iterations.

## Architecture

### Base Model
- Inherits from HuggingFace `PreTrainedModel`
- Supports any BERT-variant from Hugging Face as encoder (`bert-base-uncased`, `roberta-base`, etc.)
- Created with `AutoModel.from_pretrained()` for flexibility

### Edit Operation Head
- Predicts four operation types: KEEP, REMOVE, SPLIT, REPLACE
- Includes token replacement logits for REPLACE operations
- Includes split logits for SPLIT operations

### Recursive Controller
- Applies edit operations recursively until convergence or maximum iterations
- Tracks edit traces for RL training
- Handles tokenization and detokenization

## Training Pipeline

### Supervised Pretraining
1. Create labeled dataset with (original_text, edited_text) pairs
2. Extract ground truth edit operations
3. Train model to predict these operations with cross-entropy loss
4. Save as initialization for RL phase

### RL Training Loop
1. Sample edit operations from the model
2. Apply edits recursively until convergence
3. Compute rewards based on text quality improvement
4. Update model using policy gradient methods

## Reward Functions
- **LanguageModelPerplexity**: Fluency improvement
- **GrammarCorrectness**: Using external grammar checker
- **IterationEfficiency**: Bonus for fewer iterations
- **EditDistance**: Penalize excessive changes

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rledit.git
cd rledit

# Install the package
pip install -e .
```

## Usage

### Editing Text

```python
from transformers import AutoTokenizer
from rledit.models import BERTEditor, RecursiveEditor

# Load the model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BERTEditor.from_pretrained(model_name)

# Create the recursive editor
editor = RecursiveEditor(
    editor_model=model,
    tokenizer=tokenizer,
    max_iterations=5,
    convergence_threshold=0.95,
)

# Edit text
text = "This is a example of text with some errors."
edited_text, edit_trace = editor.edit_until_convergence(text)
print(f"Original: {text}")
print(f"Edited: {edited_text}")
```

### Training the Model

```bash
python train.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --output_dir output \
    --log_dir logs \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --max_iterations 5 \
    --convergence_threshold 0.95 \
    --temperature 1.0 \
    --discount_factor 0.95 \
    --efficiency_factor 0.1
```

### Using the Command Line Interface

```bash
python main.py \
    --model_name_or_path bert-base-uncased \
    --input_text "This is a example of text with some errors." \
    --output_file edited.txt \
    --log_file edit.log \
    --evaluate
```

## Project Structure

```
rledit/
├── models/
│   ├── bert_editor.py         # BERT-based editor model
│   ├── edit_operations.py     # Edit operations and head
│   └── recursive_editor.py    # Recursive editor controller
├── training/
│   ├── environment.py         # RL environment
│   ├── rewards.py             # Reward functions
│   └── rl_trainer.py          # RL training loop
├── data/
│   └── dataset.py             # Dataset and collator
└── utils/
    ├── tokenization.py        # Tokenization utilities
    ├── evaluation.py          # Evaluation metrics
    └── logging.py             # Logging utilities
```

## Evaluation Methods
1. Automatic metrics: BLEU, perplexity, grammar check
2. Human evaluation: pair-wise comparisons
3. Convergence rate: average iterations to stability
4. Edit efficiency: ratio of quality improvement to edits made

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@software{rledit2025,
  author = {Author},
  title = {Recursive BERT-based Text Editor with RL Training Pipeline},
  year = {2025},
  url = {https://github.com/d0rc/rledit}
}
