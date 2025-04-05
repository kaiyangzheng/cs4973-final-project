#!/usr/bin/env python3
"""
Script to fine-tune a model for paper categorization
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def prepare_data(input_file, test_size=0.2, max_samples=None):
    """
    Prepare data for fine-tuning
    
    Args:
        input_file: Path to the input file (JSON format)
        test_size: Proportion of data to use for testing
        max_samples: Maximum number of samples to use
        
    Returns:
        train_texts, train_labels, test_texts, test_labels
    """
    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Limit the number of samples if requested
    if max_samples and len(data) > max_samples:
        data = data[:max_samples]
    
    print(f"Loaded {len(data)} samples from {input_file}")
    
    # Extract texts and labels
    texts = [sample['text'] for sample in data]
    labels = [sample['labels'] for sample in data]
    
    # Split into train and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42
    )
    
    print(f"Train set: {len(train_texts)} samples")
    print(f"Test set: {len(test_texts)} samples")
    
    return train_texts, train_labels, test_texts, test_labels

def format_for_training(texts, labels, all_categories):
    """
    Format data for training
    
    Args:
        texts: List of texts
        labels: List of labels
        all_categories: Set of all categories
        
    Returns:
        List of formatted examples
    """
    examples = []
    for text, label_list in zip(texts, labels):
        # Create instruction
        instruction = "Given the following research paper, categorize it into the appropriate CS subfields."
        
        # Format the input
        input_text = text
        
        # Format the output
        output_text = ", ".join(label_list)
        
        # Create example
        examples.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })
    
    return examples

def generate_config_file(output_dir, model_name, train_file, validation_file):
    """
    Generate configuration file for fine-tuning
    
    Args:
        output_dir: Directory to save the config file
        model_name: Name of the base model
        train_file: Path to the training file
        validation_file: Path to the validation file
    """
    config = {
        "base_model": model_name,
        "model_type": "LlamaForCausalLM",
        "tokenizer_type": "LlamaTokenizer",
        "is_llama_derived_model": True,
        "load_in_8bit": False,
        "load_in_4bit": True,
        "strict": False,
        "push_to_hub": False,
        "inference_tp_size": 1,
        "training_tp_size": 1,
        "train_on_inputs": False,
        "group_by_length": True,
        "bf16": True,
        "fp16": False,
        "tf32": False,
        "gradient_checkpointing": True,
        "optim": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 0.3,
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "max_seq_length": 2048,
        "model_max_length": 2048,
        "gradient_accumulation_steps": 1,
        "micro_batch_size": 1,
        "num_epochs": 3,
        "learning_rate": 2e-4,
        "train_on_input": False,
        "dataset": {
            "path": train_file,
            "val_path": validation_file,
            "conversation_template": "chatml",
            "train_test_split": 1,
            "seed": 42
        }
    }
    
    # Save the config file
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved config file to {config_path}")
    return config_path

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Fine-tune a model for paper categorization')
    parser.add_argument('--input_file', required=True, help='Path to the input file (JSON format)')
    parser.add_argument('--output_dir', default='./data/finetune', help='Directory to save fine-tuning data')
    parser.add_argument('--model_name', default='meta-llama/Meta-Llama-3-8B-Instruct', help='Base model name')
    parser.add_argument('--max_samples', type=int, default=10000, help='Maximum number of samples to use')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare data
    train_texts, train_labels, test_texts, test_labels = prepare_data(
        args.input_file, max_samples=args.max_samples
    )
    
    # Get all categories
    all_categories = set()
    for label_list in train_labels + test_labels:
        all_categories.update(label_list)
    
    print(f"Total number of categories: {len(all_categories)}")
    
    # Format data for training
    train_examples = format_for_training(train_texts, train_labels, all_categories)
    test_examples = format_for_training(test_texts, test_labels, all_categories)
    
    # Save formatted data
    train_file = os.path.join(args.output_dir, "train.json")
    test_file = os.path.join(args.output_dir, "test.json")
    
    with open(train_file, 'w') as f:
        json.dump(train_examples, f, indent=2)
    
    with open(test_file, 'w') as f:
        json.dump(test_examples, f, indent=2)
    
    print(f"Saved train data to {train_file}")
    print(f"Saved test data to {test_file}")
    
    # Generate config file
    config_path = generate_config_file(args.output_dir, args.model_name, train_file, test_file)
    
    # Print instructions for fine-tuning
    print("\nTo fine-tune the model using Axolotl:")
    print("1. Install Axolotl: pip install axolotl")
    print(f"2. Run: accelerate launch -m axolotl.cli.train {config_path}")
    print("\nTo run this with 8-bit precision on 1 GPU with minimal requirements:")
    print(f"accelerate launch -m axolotl.cli.train {config_path} --load_in_8bit=true --push_to_hub=false")
    
if __name__ == '__main__':
    main() 