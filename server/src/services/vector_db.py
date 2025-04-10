import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pickle
import hashlib

# Simple in-memory vector database for now
# In production, you would use a proper vector database like Pinecone, Qdrant, etc.
class VectorDatabase:
    def __init__(self):
        self.vectors = []
        self.metadata = []
        self.categories = set()
        self.paper_categories = {}  # Add back paper categories dictionary
        
    def add_vector(self, vector: np.ndarray, metadata: Dict[str, Any]):
        """Add a vector and its metadata to the database"""
        self.vectors.append(vector)
        self.metadata.append(metadata)
        if 'categories' in metadata:
            self.categories.update(metadata['categories'])
            if 'id' in metadata:
                self.paper_categories[metadata['id']] = metadata['categories']
                
    def add_paper_category(self, paper_id: str, categories: List[str]):
        """Add categories for a paper"""
        self.paper_categories[paper_id] = categories
        self.categories.update(categories)
        
    def get_paper_categories(self, paper_id: str) -> List[str]:
        """Get categories for a paper"""
        return self.paper_categories.get(paper_id, [])
        
    def search(self, query_vector: np.ndarray, top_k: int = 5, boost_categories: List[str] = None, category_boost: float = 0.2) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar vectors using cosine similarity, with optional category boosting
        
        Args:
            query_vector: The query vector to search for
            top_k: Number of results to return
            boost_categories: Categories to boost in the search results
            category_boost: How much to boost papers with matching categories
            
        Returns:
            List of (metadata, similarity) tuples
        """
        if not self.vectors:
            return []
            
        # Convert query vector to numpy array if it isn't already
        query_vector = np.array(query_vector)
        
        # Ensure query vector has the same dimensions as database vectors
        if len(query_vector) != len(self.vectors[0]):
            # Pad or truncate query vector to match database vector dimensions
            if len(query_vector) < len(self.vectors[0]):
                query_vector = np.pad(query_vector, (0, len(self.vectors[0]) - len(query_vector)))
            else:
                query_vector = query_vector[:len(self.vectors[0])]
        
        # Calculate cosine similarities
        similarities = np.dot(self.vectors, query_vector)
        
        # Apply category boosting if categories are provided
        if boost_categories and len(boost_categories) > 0:
            print(f"Boosting papers with categories: {boost_categories}")
            for i, metadata in enumerate(self.metadata):
                paper_id = metadata.get('id')
                if paper_id and paper_id in self.paper_categories:
                    paper_cats = self.paper_categories[paper_id]
                    # Add boost for each matching category
                    matching_categories = set(boost_categories).intersection(set(paper_cats))
                    if matching_categories:
                        # Boost proportional to number of matching categories
                        boost = category_boost * len(matching_categories) / len(boost_categories)
                        similarities[i] += boost
                        print(f"Boosting paper {metadata.get('title', 'Unknown')} by {boost} for matching categories: {matching_categories}")
        
        # Get top k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.metadata[i], float(similarities[i])) for i in top_indices]
        
    def get_categories(self) -> List[str]:
        """Get all unique categories in the database"""
        return list(self.categories)
        
    def save_embeddings(self, filename: str):
        """Save embeddings and metadata to a file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'vectors': self.vectors,
                'metadata': self.metadata,
                'categories': list(self.categories),
                'paper_categories': self.paper_categories
            }, f)
            
    def load_embeddings(self, filename: str):
        """Load embeddings and metadata from a file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.vectors = data['vectors']
                self.metadata = data['metadata']
                self.categories = set(data['categories'])
                self.paper_categories = data.get('paper_categories', {})

# Create a singleton instance
vector_db = VectorDatabase()

def get_embedding(text: str) -> np.ndarray:
    """Generate a simple hash-based embedding for the text"""
    # Generate a hash of the text
    hash_obj = hashlib.sha256(text.encode())
    hash_hex = hash_obj.hexdigest()
    
    # Convert hash to a numpy array of floats
    # Use first 1536 bits (192 bytes) of the hash
    hash_bytes = bytes.fromhex(hash_hex)[:192]
    embedding = np.frombuffer(hash_bytes, dtype=np.float32)
    
    # Pad or truncate to 1536 dimensions if needed
    if len(embedding) < 1536:
        embedding = np.pad(embedding, (0, 1536 - len(embedding)))
    else:
        embedding = embedding[:1536]
    
    # Normalize the embedding
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding

async def find_related_papers(text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Find related papers based on text input
    
    Args:
        text: The text to find related papers for
        top_k: Number of related papers to return
        
    Returns:
        List of related papers with metadata
    """
    # Get embedding for the text
    embedding = get_embedding(text)
    
    # Search for similar papers
    similar_papers = vector_db.search(embedding, top_k=top_k)
    
    # Extract metadata
    return [paper for paper, _ in similar_papers]

def categorize_paper(paper_content: str) -> List[str]:
    """
    Categorize a paper into relevant research areas
    
    Args:
        paper_content: The content of the paper
        
    Returns:
        List of categories
    """
    # This would use a fine-tuned model in production
    # For now, we'll use a simple approach
    categories = [
        "Machine Learning",
        "Natural Language Processing",
        "Computer Vision",
        "Robotics",
        "Systems",
        "Theory",
        "Security",
        "Networks",
        "Databases",
        "Human-Computer Interaction"
    ]
    
    # Count occurrences of category-related terms
    category_scores = {category: 0 for category in categories}
    
    # Simple keyword matching
    keywords = {
        "Machine Learning": ["learning", "neural", "network", "model", "training"],
        "Natural Language Processing": ["language", "nlp", "text", "word", "sentence"],
        "Computer Vision": ["vision", "image", "visual", "recognition", "detection"],
        "Robotics": ["robot", "robotic", "motion", "control", "manipulation"],
        "Systems": ["system", "performance", "scalability", "distributed", "parallel"],
        "Theory": ["proof", "theorem", "complexity", "algorithm", "mathematical"],
        "Security": ["security", "privacy", "encryption", "attack", "vulnerability"],
        "Networks": ["network", "protocol", "routing", "bandwidth", "latency"],
        "Databases": ["database", "query", "index", "transaction", "storage"],
        "Human-Computer Interaction": ["interface", "user", "interaction", "design", "usability"]
    }
    
    # Score each category
    for category, terms in keywords.items():
        for term in terms:
            if term.lower() in paper_content.lower():
                category_scores[category] += 1
    
    # Get top 3 categories
    top_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    return [category for category, _ in top_categories] 