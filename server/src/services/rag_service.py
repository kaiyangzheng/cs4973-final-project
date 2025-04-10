import os
from typing import List, Dict, Any, Optional
import numpy as np
from .vector_db import vector_db, get_embedding
from .llm_client import call_llm
from .model_service import model_service  # Import model service for category labels

class RAGService:
    """
    A service that implements Retrieval-Augmented Generation for research papers.
    It enhances responses by finding and incorporating relevant context from the paper database.
    """
    
    def __init__(self):
        self.context_window = 2000  # Number of tokens to use for context
    
    async def find_relevant_context(self, query: str, paper_content: Optional[str] = None, paper_categories: List[str] = None) -> List[Dict[str, Any]]:
        """
        Find relevant context from the paper database
        
        Args:
            query: The user's query
            paper_content: Optional content of the current paper
            paper_categories: Optional categories of the current paper for boosting similar papers
            
        Returns:
            List of context snippets from relevant papers
        """
        try:
            # Get embedding for the query
            query_embedding = get_embedding(query)
            
            # Get paper categories if not provided but paper content is available
            if not paper_categories and paper_content:
                try:
                    paper_categories = model_service.predict_categories(paper_content)
                    print(f"Predicted paper categories for search boosting: {paper_categories}")
                except Exception as e:
                    print(f"Error predicting paper categories: {str(e)}")
                    paper_categories = []
            
            # Search for similar papers with category boosting if categories are available
            similar_papers = vector_db.search(
                query_embedding, 
                top_k=5,  # Increase to improve chances of finding relevant papers
                boost_categories=paper_categories
            )
            
            # Filter out papers with zero similarity
            similar_papers = [(paper, sim) for paper, sim in similar_papers if sim > 0]
            
            if not similar_papers:
                print("No papers with non-zero similarity found")
                
                # Alternative approach: try searching without categories
                if paper_categories:
                    print("Trying search without category boosting...")
                    similar_papers = vector_db.search(
                        query_embedding, 
                        top_k=10,
                        boost_categories=None
                    )
                    similar_papers = [(paper, sim) for paper, sim in similar_papers if sim > 0]
                    
                # If still no results, try a broader search approach
                if not similar_papers:
                    print("No papers found. Using fallback method to find any papers...")
                    similar_papers = vector_db.search(
                        query_embedding, 
                        top_k=5,
                        boost_categories=None
                    )
            
            # Extract relevant sections from each paper
            context_snippets = []
            for paper, similarity in similar_papers:
                print(f"Processing paper with similarity {similarity}: {paper.get('title', 'Unknown')}")
                snippets = self._extract_relevant_sections(paper, query)
                if snippets:
                    # Include paper categories if available
                    paper_id = paper.get('id')
                    paper_cats = vector_db.get_paper_categories(paper_id) if paper_id else []
                    
                    context_snippets.append({
                        'title': paper.get('title', 'Unknown Paper'),
                        'snippets': snippets,
                        'categories': paper_cats,  # Include paper categories
                        'similarity': similarity   # Include similarity score
                    })
                else:
                    print(f"No relevant snippets found in paper: {paper.get('title', 'Unknown')}")
            
            if not context_snippets:
                print("No context snippets found in any paper")
                
            return context_snippets
            
        except Exception as e:
            print(f"Error finding relevant context: {str(e)}")
            import traceback
            traceback.print_exc()
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
        try:
            # Handle different paper data structures
            if isinstance(paper, dict):
                # Check if paper has 'content' key (direct structure)
                if 'content' in paper:
                    # Direct content structure - extract from content field
                    content = paper.get('content', '')
                    if not content:
                        print(f"Warning: Paper content is empty for paper: {paper.get('title', 'Unknown')}")
                        return []
                        
                    # Create a single section with the content
                    return [{
                        "section": "content",
                        "content": content[:self.context_window],
                        "paper_title": paper.get("title", "Untitled"),
                        "paper_authors": paper.get("authors", "Unknown"),
                        "paper_year": paper.get("year", ""),
                        "paper_url": paper.get("url", "")
                    }]
                
                # Otherwise use the section-based approach
                # Extract key sections (abstract, introduction, conclusion)
                sections = {
                    "abstract": paper.get("abstract", ""),
                    "introduction": paper.get("introduction", ""),
                    "conclusion": paper.get("conclusion", "")
                }
                
                # Make sure query is well-formed
                query_terms = [term.lower() for term in query.split() if term]
                if not query_terms:
                    query_terms = ["the"]  # Fallback to ensure we get some results
                
                # Find sections containing query terms
                relevant_sections = []
                for section_name, content in sections.items():
                    if not content:
                        continue
                        
                    try:
                        # Check if any query term is in the content
                        if any(term in content.lower() for term in query_terms):
                            relevant_sections.append({
                                "section": section_name,
                                "content": content[:self.context_window],  # Limit context length
                                "paper_title": paper.get("title", "Untitled"),
                                "paper_authors": paper.get("authors", "Unknown"),
                                "paper_year": paper.get("year", ""),
                                "paper_url": paper.get("url", "")
                            })
                    except Exception as e:
                        print(f"Error processing section {section_name}: {str(e)}")
                
                return relevant_sections
            else:
                print(f"Warning: Paper is not a dictionary: {type(paper)}")
                return []
                
        except Exception as e:
            print(f"Error extracting relevant sections: {str(e)}")
            return []
    
    async def generate_response(self, query: str, paper_content: Optional[str] = None, use_rag: bool = True) -> str:
        """
        Generate a response using RAG
        
        Args:
            query: The user's query
            paper_content: Optional content of the current paper
            
        Returns:
            Generated response with relevant context
        """
        print(f"Generating response for query: {query}")
        print(f"Current paper content: {paper_content[:100] if paper_content else 'No content provided'}")
        print(f"Use RAG: {use_rag}")
        
        # Get categories for the current paper if available
        paper_categories = []
        if paper_content and use_rag:
            try:
                paper_categories = model_service.predict_categories(paper_content)
                print(f"Current paper categories: {paper_categories}")
            except Exception as e:
                print(f"Error getting paper categories: {str(e)}")
        
        # Find relevant context with category boosting
        context_snippets = await self.find_relevant_context(query, paper_content, paper_categories)
        print("context snippets:")
        print(context_snippets)

        # Format context for the prompt
        context_prompt = ""
        if context_snippets and use_rag:
            context_prompt = "\n\nRelevant research context:\n\n"
            for i, snippet in enumerate(context_snippets, 1):
                # Add categories for the paper if available
                categories_str = ""
                if snippet.get('categories'):
                    # Try to get human-readable labels for categories
                    try:
                        category_labels = [
                            f"{cat} ({model_service.get_category_label(cat)})" 
                            for cat in snippet.get('categories', [])
                        ]
                        categories_str = f" [Categories: {', '.join(category_labels)}]"
                    except Exception:
                        # Fall back to just the category codes if labels aren't available
                        categories_str = f" [Categories: {', '.join(snippet.get('categories', []))}]"
                
                context_prompt += f"{i}. From {snippet['title']} ({snippet.get('paper_year', 'Unknown')}){categories_str}:\n"
                for snip in snippet['snippets']:
                    context_prompt += f"   {snip['section'].upper()}: {snip['content']}\n\n"
        
        # Add current paper content and categories if available
        paper_prompt = ""
        if paper_content:
            paper_prompt = f"\n\nCurrent paper content:\n\n{paper_content[:self.context_window]}"
            
            # Add current paper categories if available
            if paper_categories and use_rag:
                try:
                    # Try to get human-readable labels for categories
                    category_labels = [
                        f"{cat} ({model_service.get_category_label(cat)})" 
                        for cat in paper_categories
                    ]
                    paper_prompt += f"\n\nCurrent paper categories: {', '.join(category_labels)}"
                except Exception:
                    # Fall back to just the category codes
                    paper_prompt += f"\n\nCurrent paper categories: {', '.join(paper_categories)}"
        

        # Generate response using the LLM
        if use_rag:
            full_prompt = f"""
            Based on the following context and query:
            
            {context_prompt}
            {paper_prompt}
            
            Query: {query}
            """
            print("RAG prompt: " + full_prompt)
        else:
            full_prompt = f"""
            Based on the following query:
            
            {paper_prompt}
            
            Query: {query}
            """        
            print("Non-RAG prompt: " + full_prompt)
        response = await call_llm(full_prompt)
        return response

# Create singleton instance
rag_service = RAGService() 