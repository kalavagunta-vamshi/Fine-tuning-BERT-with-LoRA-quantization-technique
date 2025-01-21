# Fine-Tuned BERT for Movie Review Classification

This repository contains a Python implementation for fine-tuning a BERT-based model to classify movie reviews into positive or negative sentiment. The code leverages Hugging Face's `transformers` library, along with `datasets`, `evaluate`, and `peft` for efficient model training and evaluation.

## Features

- **Dataset:** IMDB dataset is used for training and validation.
- **Model:** Fine-tuned `distilbert-base-uncased` model.
- **Parameter-Efficient Fine-Tuning:** LoRA (Low-Rank Adaptation) is applied to optimize memory and computational efficiency.
- **Metrics:** Accuracy metric is calculated to evaluate model performance.

## Installation

To set up the environment, install the required packages using:

```bash
pip install datasets evaluate transformers[sentencepiece]
pip install accelerate -U
pip install peft
```

## Dataset

The IMDB dataset is used, which contains 50,000 movie reviews categorized into positive and negative sentiments.

## Model Architecture

- **Base Model:** `distilbert-base-uncased`
- **Label Mapping:** 
  - `0`: Negative
  - `1`: Positive
- **Fine-Tuning Method:** LoRA with configurations:
  - `r=4`
  - `lora_alpha=32`
  - `lora_dropout=0.01`

## Code Workflow

1. **Dataset Preparation:**
   - Load and preprocess IMDB dataset.
   - Tokenize text data using `AutoTokenizer`.
   - Prepare training and validation splits with 500 samples each.

2. **Model Initialization:**
   - Load pre-trained DistilBERT model for sequence classification.
   - Add LoRA layers for efficient fine-tuning.

3. **Training:**
   - Define training arguments using `TrainingArguments`.
   - Train the model using Hugging Face's `Trainer` class.

4. **Evaluation:**
   - Evaluate the model on a validation set.
   - Compute accuracy using the `evaluate` library.

5. **Predictions:**
   - Test the model on example movie reviews and output predictions.

## Usage

Run the script to train and evaluate the model:

```python
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model
import evaluate
import numpy as np
import torch
```

Use the following sentiment examples to test the model:

```python
sentiment_examples = [
    "The movie was an absolute delight from start to finish!",
    "I wouldn't recommend this film to anyone; it was quite a letdown.",
    ...
]
```

## Example Output

### Before Training:
```
The film's innovative cinematography left me in awe throughout. - Negative
A disappointing sequel that fails to capture the magic of the original. - Negative
The lead actor's performance was nothing short of Oscar-worthy. - Negative
This movie is a complete disaster; avoid at all costs. - Negative
...
```

### After Training:
```
The film's innovative cinematography left me in awe throughout. - Positive
A disappointing sequel that fails to capture the magic of the original. - Negative
The lead actor's performance was nothing short of Oscar-worthy. - Positive
This movie is a complete disaster; avoid at all costs. - Negative
...
```

## LoRA Configuration

```python
peft_config = LoraConfig(
    task_type="SEQ_CLS",
    r=4,
    lora_alpha=32,
    lora_dropout=0.01,
    target_modules=['q_lin']
)
```

## Model Training Parameters

- **Learning Rate:** 1e-3
- **Batch Size:** 4
- **Epochs:** 1

## Results

Achieved high accuracy on the validation dataset with minimal computational resources.

## Acknowledgments

This project uses Hugging Face's `transformers`, `datasets`, and `evaluate` libraries along with `peft` for LoRA-based fine-tuning.
