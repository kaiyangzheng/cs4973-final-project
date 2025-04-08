import os
import json
import torch
import numpy as np
from typing import List, Dict, Any
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Check if trained model exists
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/trained_model')
MODEL_AVAILABLE = os.path.exists(MODEL_DIR) and (
    os.path.exists(os.path.join(MODEL_DIR, 'model.safetensors')) or 
    os.path.exists(os.path.join(MODEL_DIR, 'pytorch_model.bin'))
)

# Default categories as a fallback
CATEGORIES = [
    "cs.LG",  # Machine Learning
    "cs.AI",  # Artificial Intelligence
    "cs.CV",  # Computer Vision
    "cs.CL",  # Computational Linguistics
    "cs.CR",  # Cryptography and Security
    "cs.DS",  # Data Structures and Algorithms
    "cs.DB",  # Databases
    "cs.NI",  # Networking and Internet Architecture
    "cs.SE",  # Software Engineering
    "cs.HC"   # Human-Computer Interaction
]

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

# Try to load categories from model directory if available
if MODEL_AVAILABLE:
    category_mapping_path = os.path.join(MODEL_DIR, 'category_mapping.json')
    if os.path.exists(category_mapping_path):
        try:
            with open(category_mapping_path, 'r') as f:
                CATEGORIES = json.load(f)
                print(f"Loaded {len(CATEGORIES)} categories from trained model")
        except Exception as e:
            print(f"Error loading category mapping: {str(e)}")
    
    # Try to load category labels
    category_labels_path = os.path.join(MODEL_DIR, 'category_labels.json')
    if os.path.exists(category_labels_path):
        try:
            with open(category_labels_path, 'r') as f:
                loaded_labels = json.load(f)
                # Update the default labels with the loaded ones
                CATEGORY_LABELS.update(loaded_labels)
                print(f"Loaded {len(loaded_labels)} category labels from trained model")
        except Exception as e:
            print(f"Error loading category labels: {str(e)}")

class ModelService:
    """Service for paper categorization using the trained model"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialized = False
        
        # Initialize model if available
        if MODEL_AVAILABLE:
            self._initialize_model()
        else:
            print("Trained model not found. Falling back to rule-based categorization.")
    
    def _initialize_model(self):
        """Initialize the model and tokenizer"""
        try:
            print(f"Loading model from {MODEL_DIR}...")
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            self.model.to(self.device)
            self.model.eval()
            self.initialized = True
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.initialized = False
    
    def predict_categories(self, text: str, threshold: float = 0.5) -> List[str]:
        """
        Predict categories for the given text
        
        Args:
            text: The text to categorize
            threshold: Probability threshold for positive classification
            
        Returns:
            List of predicted categories
        """
        if not self.initialized:
            # Fallback to the existing rule-based categorization
            from .vector_db import categorize_paper
            return categorize_paper(text)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
            
            # Move inputs to device
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            
            # Get categories above threshold
            predicted_categories = [CATEGORIES[i] for i, prob in enumerate(probabilities) if prob >= threshold]
            
            # If no categories are above threshold, take the top 3
            if not predicted_categories:
                top_indices = np.argsort(probabilities)[-3:][::-1]
                predicted_categories = [CATEGORIES[i] for i in top_indices]
            
            return predicted_categories
            
        except Exception as e:
            print(f"Error predicting categories: {str(e)}")
            # Fallback to the existing rule-based categorization
            from .vector_db import categorize_paper
            return categorize_paper(text)
    
    def get_categories(self) -> List[str]:
        """Get all available categories"""
        return CATEGORIES
    
    def model_available(self) -> bool:
        """Check if the trained model is available"""
        return self.initialized
        
    def get_category_label(self, category: str) -> str:
        """
        Get the human-readable label for a category
        
        Args:
            category: The category code (e.g., "cs.LG")
            
        Returns:
            Human-readable label or the original category if not found
        """
        return CATEGORY_LABELS.get(category, category)
    
    def get_categories_with_labels(self) -> List[Dict[str, str]]:
        """
        Get all available categories with their human-readable labels
        
        Returns:
            List of dictionaries with 'code' and 'label' keys
        """
        return [
            {"code": category, "label": self.get_category_label(category)}
            for category in CATEGORIES
        ]
    
    def predict_categories_with_labels(self, text: str, threshold: float = 0.5) -> List[Dict[str, str]]:
        """
        Predict categories with labels for the given text
        
        Args:
            text: The text to categorize
            threshold: Probability threshold for positive classification
            
        Returns:
            List of dictionaries with 'code' and 'label' keys for predicted categories
        """
        categories = self.predict_categories(text, threshold)
        return [
            {"code": category, "label": self.get_category_label(category)}
            for category in categories
        ]

# Create singleton instance
model_service = ModelService() 