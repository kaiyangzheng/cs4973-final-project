import os
from typing import List, Dict, Any, Optional
import numpy as np
from .vector_db import vector_db, get_embedding
from .llm_client import call_llm

class RAGService:
    """
    A service that implements Retrieval-Augmented Generation for research papers.
    It enhances responses by finding and incorporating relevant context from the paper database.
    """
    
    def __init__(self):
        self.context_window = 2000  # Number of tokens to use for context
    
    async def find_relevant_context(self, query: str, paper_content: str) -> List[Dict[str, Any]]:
        """Find relevant context from the paper database"""
        try:
            # Get embedding for the query
            query_embedding = get_embedding(query)  # Remove await since it's synchronous
            
            # Search for similar papers
            similar_papers = vector_db.search(query_embedding, top_k=5)
            
            # Extract relevant sections from each paper
            context_snippets = []
            for paper, _ in similar_papers:
                if 'content' in paper:
                    snippets = self._extract_relevant_sections(paper['content'], query)
                    if snippets:
                        context_snippets.append({
                            'title': paper.get('title', 'Unknown Paper'),
                            'snippets': snippets
                        })
            
            return context_snippets
            
        except Exception as e:
            print(f"Error finding relevant context: {str(e)}")
            return []
    
    def _extract_relevant_sections(self, paper: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """
        Extract relevant sections from a paper based on the query
        
        Args:
            paper: Paper metadata and content
            query: User's query
            
        Returns:
            List of relevant sections with metadata
        """
        # This is a simplified implementation
        # In a real system, you would use more sophisticated NLP techniques
        
        # Extract key sections (abstract, introduction, conclusion)
        sections = {
            "abstract": paper.get("abstract", ""),
            "introduction": paper.get("introduction", ""),
            "conclusion": paper.get("conclusion", "")
        }
        
        # Find sections containing query terms
        relevant_sections = []
        for section_name, content in sections.items():
            if content and any(term.lower() in content.lower() for term in query.split()):
                relevant_sections.append({
                    "section": section_name,
                    "content": content[:self.context_window],  # Limit context length
                    "paper_title": paper.get("title", "Untitled"),
                    "paper_authors": paper.get("authors", "Unknown"),
                    "paper_year": paper.get("year", ""),
                    "paper_url": paper.get("url", "")
                })
        
        return relevant_sections
    
    async def generate_response(self, query: str, paper_content: Optional[str] = None) -> str:
        """
        Generate a response using RAG
        
        Args:
            query: The user's query
            paper_content: Optional content of the current paper
            
        Returns:
            Generated response with relevant context
        """
        # Find relevant context
        context_snippets = await self.find_relevant_context(query, paper_content)
        
        # Format context for the prompt
        context_prompt = ""
        if context_snippets:
            context_prompt = "\n\nRelevant research context:\n\n"
            for i, snippet in enumerate(context_snippets, 1):
                context_prompt += f"{i}. From {snippet['paper_title']} ({snippet['paper_year']}):\n"
                context_prompt += f"{snippet['content']}\n\n"
        
        # Add current paper content if available
        paper_prompt = ""
        if paper_content:
            paper_prompt = f"\n\nCurrent paper content:\n\n{paper_content[:self.context_window]}"
        
        # Generate response using the LLM
        full_prompt = f"""
        Based on the following context and query:
        
        {context_prompt}
        {paper_prompt}
        
        Query: {query}
        """
        
        response = await call_llm(full_prompt)
        return response

# Create singleton instance
rag_service = RAGService() 