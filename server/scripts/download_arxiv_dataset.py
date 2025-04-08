#!/usr/bin/env python3
"""
Script to download and process the ArXiv CS papers dataset for fine-tuning
Dataset from: https://www.kaggle.com/datasets/devintheai/arxiv-cs-papers-multi-label-classification-200k-v1/data
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import zipfile
import requests

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/devintheai/arxiv-cs-papers-multi-label-classification-200k-v1/download?datasetVersionNumber=1"

def download_dataset(output_dir):
    """
    Download the ArXiv CS papers dataset from Kaggle
    
    Args:
        output_dir: Directory to save the dataset
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Kaggle requires authentication, so we'll just print instructions
    print("To download the dataset from Kaggle:")
    print(f"1. Go to {KAGGLE_DATASET_URL}")
    print("2. Sign in to Kaggle (or create an account)")
    print("3. Click the 'Download' button")
    print(f"4. Move the downloaded zip file to {output_dir}")
    print("5. Run this script again with --extract_only flag")
    
    return False

def extract_dataset(output_dir, zip_path=None):
    """
    Extract the ArXiv CS papers dataset
    
    Args:
        output_dir: Directory to extract the dataset to
        zip_path: Path to the zip file
    """
    # Find the zip file if not provided
    if not zip_path:
        zip_files = [f for f in os.listdir(output_dir) if f.endswith('.zip')]
        if not zip_files:
            print("No zip files found in the output directory.")
            return False
        zip_path = os.path.join(output_dir, zip_files[0])
    
    # Extract the zip file
    print(f"Extracting {zip_path} to {output_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    return True

def process_dataset(output_dir):
    """
    Process the ArXiv CS papers dataset for fine-tuning
    
    Args:
        output_dir: Directory containing the extracted dataset
    """
    # Look for CSV files
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the output directory.")
        return False
    
    # Process each CSV file
    for csv_file in csv_files:
        csv_path = os.path.join(output_dir, csv_file)
        print(f"Processing {csv_path}...")
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} papers")
        
        # Print some statistics
        print(f"Number of papers: {len(df)}")
        if 'categories' in df.columns:
            all_categories = []
            for cats in df['categories'].dropna():
                if isinstance(cats, str):
                    all_categories.extend(cats.split())
            unique_categories = set(all_categories)
            print(f"Number of unique categories: {len(unique_categories)}")
            print(f"Top 10 categories: {pd.Series(all_categories).value_counts().head(10).to_dict()}")
        
        # Create a format suitable for fine-tuning
        output_format = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            title = row.get('title', '')
            abstract = row.get('abstract', '')
            categories = row.get('categories', '').split() if isinstance(row.get('categories', ''), str) else []
            
            # Skip papers without categories or abstract
            if not categories or not abstract:
                continue
            
            # Format for fine-tuning
            output_format.append({
                'text': f"Title: {title}\n\nAbstract: {abstract}",
                'labels': categories
            })
        
        # Save the processed dataset
        output_path = os.path.join(output_dir, f"processed_{csv_file.replace('.csv', '.json')}")
        with open(output_path, 'w') as f:
            json.dump(output_format, f, indent=2)
        
        print(f"Saved processed dataset to {output_path}")
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Download and process the ArXiv CS papers dataset')
    parser.add_argument('--output_dir', default='./data/arxiv_cs_papers', help='Directory to save the dataset')
    parser.add_argument('--extract_only', action='store_true', help='Extract the dataset only')
    parser.add_argument('--process_only', action='store_true', help='Process the dataset only')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract the dataset if requested
    if args.extract_only:
        extract_dataset(args.output_dir)
        return
    
    # Process the dataset if requested
    if args.process_only:
        process_dataset(args.output_dir)
        return
    
    # Otherwise, do the full pipeline
    download_dataset(args.output_dir)
    
if __name__ == '__main__':
    main() 