#!/usr/bin/env python3
"""
Script to train a model for paper categorization using the existing codebase.
Integrates with the current RAG implementation.
"""

import os
import sys
import json
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from dotenv import load_dotenv

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# Import services from the existing codebase
from src.services.vector_db import categorize_paper, get_embedding, vector_db

# Initialize categories as empty list, will be populated from the dataset
CATEGORIES = []

# Category labels mapping
CATEGORY_LABELS = {
    "cs.LG": "Machine Learning",
    "cs.AI": "Artificial Intelligence",
    "cs.CV": "Computer Vision",
    "cs.CL": "Computational Linguistics",
    "cs.CR": "Cryptography and Security",
    "cs.DS": "Data Structures and Algorithms",
    "cs.DB": "Databases",
    "cs.NI": "Networking and Internet Architecture",
    "cs.SE": "Software Engineering",
    "cs.HC": "Human-Computer Interaction",
    # Common categories not in the default list but might be in the dataset
    "cs.CY": "Computers and Society",
    "cs.DC": "Distributed Computing",
    "cs.GT": "Computer Science and Game Theory",
    "cs.IR": "Information Retrieval",
    "cs.IT": "Information Theory",
    "cs.MM": "Multimedia",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.PL": "Programming Languages",
    "cs.RO": "Robotics",
    "cs.SI": "Social and Information Networks",
    "cs.SY": "Systems and Control",
}

# Simple implementation of categorize_paper function directly in this script
def categorize_paper(paper_content: str) -> list:
    """
    Categorize a paper into relevant research areas
    
    Args:
        paper_content: The content of the paper
        
    Returns:
        List of categories
    """
    # Use placeholder if categories haven't been loaded yet
    if not CATEGORIES:
        return ["cs.LG", "cs.AI", "cs.CV"]
        
    # Count occurrences of category-related terms
    category_scores = {category: 0 for category in CATEGORIES}
    
    # Simple keyword matching - map common terms to categories
    keywords = {}
    
    # Add some basic keywords for common CS categories
    # This is just a fallback and will be less important once we extract real categories
    common_keywords = {
        "cs.LG": ["learning", "neural", "network", "model", "training"],
        "cs.AI": ["intelligence", "ai", "reasoning", "cognitive", "agent"],
        "cs.CV": ["vision", "image", "visual", "recognition", "detection"],
        "cs.CL": ["language", "nlp", "text", "word", "sentence"],
        "cs.CR": ["security", "privacy", "encryption", "attack", "vulnerability"],
        "cs.DS": ["algorithm", "structure", "complexity", "optimization", "graph"],
        "cs.DB": ["database", "query", "index", "transaction", "storage"],
        "cs.NI": ["network", "protocol", "routing", "bandwidth", "latency"],
        "cs.SE": ["software", "development", "testing", "engineering", "design"],
        "cs.HC": ["interface", "user", "interaction", "design", "usability"]
    }
    
    # Add common keywords for categories we know about
    for category in CATEGORIES:
        if category in common_keywords:
            keywords[category] = common_keywords[category]
        else:
            # For other categories, use the category name itself as a keyword
            keywords[category] = [category.lower().replace("cs.", "")]
    
    # Score each category based on keyword matches
    for category, terms in keywords.items():
        for term in terms:
            if term.lower() in paper_content.lower():
                category_scores[category] += 1
    
    # Get top 3 categories
    top_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    return [category for category, _ in top_categories if _ > 0]

class PaperDataset(Dataset):
    """Dataset for paper classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        for key in encoding:
            encoding[key] = encoding[key].squeeze(0)
        
        # Add labels
        encoding["labels"] = torch.tensor(label, dtype=torch.float)
        
        return encoding

def process_csv_to_dataset(csv_path, test_size=0.2, max_samples=None):
    """
    Process CSV file into dataset
    
    Args:
        csv_path: Path to the CSV file
        test_size: Proportion of data for testing
        max_samples: Maximum number of samples to use
        
    Returns:
        train_texts, train_labels, test_texts, test_labels
    """
    print(f"Processing {csv_path}...")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} papers")
    
    # Print some statistics
    print(f"Number of papers: {len(df)}")
    
    # Extract all unique categories from the dataset
    global CATEGORIES
    if 'categories' in df.columns:
        all_categories = []
        for cats in df['categories'].dropna():
            if isinstance(cats, str):
                all_categories.extend(cats.split())
        
        # Get unique categories and sort them
        CATEGORIES = sorted(list(set(all_categories)))
        print(f"Number of unique categories: {len(CATEGORIES)}")
        print(f"Categories: {CATEGORIES[:10]}... (showing first 10)")
        
        # Print category statistics
        cat_counts = pd.Series(all_categories).value_counts()
        print(f"Top 10 categories: {cat_counts.head(10).to_dict()}")
    else:
        print("Warning: No 'categories' column found in the dataset")
        CATEGORIES = ["cs.LG", "cs.AI", "cs.CV"]
    
    # Prepare data
    texts = []
    labels = []
    
    # Limit the number of samples if requested
    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)
        print(f"Limited to {max_samples} samples")
    
    # Process each paper
    for _, row in tqdm(df.iterrows(), total=len(df)):
        title = row.get('title', '')
        abstract = row.get('abstract', '')
        
        # Skip papers without title or abstract
        if not title or not abstract:
            continue
        
        # Get categories from the dataset
        paper_content = f"Title: {title}\n\nAbstract: {abstract}"
        if 'categories' in df.columns and isinstance(row.get('categories'), str):
            # Use the actual categories from the dataset
            categories = row.get('categories', '').split()
        else:
            # Fallback to rule-based categorization
            categories = categorize_paper(paper_content)
        
        # Convert categories to one-hot encoding
        label = [1 if cat in categories else 0 for cat in CATEGORIES]
        
        texts.append(paper_content)
        labels.append(label)
    
    print(f"Processed {len(texts)} papers")
    
    # Split into train and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42
    )
    
    print(f"Train set: {len(train_texts)} samples")
    print(f"Test set: {len(test_texts)} samples")
    
    return train_texts, train_labels, test_texts, test_labels

def train_model(train_dataset, eval_dataset, model_name, output_dir="./trained_model", batch_size=4, learning_rate=2e-5, epochs=3):
    """
    Train the model
    
    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        model_name: Base model name
        output_dir: Directory to save the model
        batch_size: Batch size
        learning_rate: Learning rate
        epochs: Number of epochs
    """
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(CATEGORIES),
        problem_type="multi_label_classification"
    )
    model.to(device)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        learning_rate=learning_rate,
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to {output_dir}")
    
    # Evaluate
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    return model

def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description='Train a model for paper categorization')
    parser.add_argument('--csv_path', default='./data/cs_papers_api.csv', help='Path to the CSV file')
    parser.add_argument('--model_name', default='distilbert-base-uncased', help='Base model name')
    parser.add_argument('--output_dir', default='./data/trained_model', help='Directory to save the model')
    parser.add_argument('--max_samples', type=int, default=10000, help='Maximum number of samples to use')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer from the selected model
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Process data
    train_texts, train_labels, test_texts, test_labels = process_csv_to_dataset(
        args.csv_path, max_samples=args.max_samples
    )
    
    # Create datasets
    train_dataset = PaperDataset(train_texts, train_labels, tokenizer)
    eval_dataset = PaperDataset(test_texts, test_labels, tokenizer)
    
    # Train model
    model = train_model(
        train_dataset,
        eval_dataset,
        args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    # Save mapping of categories to indices
    with open(os.path.join(args.output_dir, "category_mapping.json"), "w") as f:
        json.dump(CATEGORIES, f)
    
    # Create and save category labels mapping
    category_labels = {}
    for category in CATEGORIES:
        # Use existing label if available, otherwise use the category itself
        category_labels[category] = CATEGORY_LABELS.get(category, category)
    
    # Save category labels
    with open(os.path.join(args.output_dir, "category_labels.json"), "w") as f:
        json.dump(category_labels, f)
    
    print("Training complete!")
    print(f"Model saved to {args.output_dir}")
    print("\nTo use the trained model:")
    print("1. Load the model: model = AutoModelForSequenceClassification.from_pretrained('path/to/model')")
    print("2. Categorize papers: categories = predict_categories(model, tokenizer, 'Paper text')")

if __name__ == '__main__':
    main() 