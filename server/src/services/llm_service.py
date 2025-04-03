import os
import asyncio
from typing import Dict, List, Optional, Any
from .pdf_service import extract_text_from_pdf_base64, is_pdf_content, extract_metadata_from_pdf
import re
from .rag_service import rag_service
from .llm_client import call_llm

class CodeAgent:
    """
    A code agent that uses LLM to reason about and understand research papers.
    It implements deep research capabilities by categorizing papers and performing
    cross-referencing to provide better context.
    """
    
    def __init__(self):
        self.paper_categories = []
        self.context = {}
    
    async def process_query(self, prompt: str, paper_content: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query about a research paper
        
        Args:
            prompt: The user's question
            paper_content: The content of the paper being analyzed
            
        Returns:
            Dict containing the response and any other relevant data
        """
        print(f"\n==== process_query called with prompt: {prompt[:50]}... ====")
        try:
            # Process paper content if provided
            paper_text = paper_content
            paper_metadata = None
            paper_categories = []
            
            if paper_text:
                print(f"Processing paper content of length: {len(paper_text)}")
                
                # Categorize the paper
                try:
                    from src.services.vector_db import categorize_paper, vector_db
                    
                    # Generate a unique ID based on content hash
                    import hashlib
                    paper_id = hashlib.md5(paper_text[:5000].encode('utf-8')).hexdigest()
                    
                    # Categorize the paper
                    print("Categorizing paper...")
                    paper_categories = categorize_paper(paper_text)  # Remove await since it's synchronous
                    print(f"Paper categories: {paper_categories}")  
                    
                    # Add categories to vector database
                    vector_db.add_paper_category(paper_id, paper_categories)
                    
                except Exception as e:
                    print(f"Error categorizing paper: {str(e)}")
                    paper_categories = []
            
            # Generate response using RAG
            print("Generating response using RAG...")
            response = await rag_service.generate_response(prompt, paper_text)
            print(f"RAG response received, length: {len(response)}")
            
            # Prepare the final response
            result = {
                "success": True,
                "response": response,
                "pending": False
            }
            
            # Add paper information if available
            if paper_categories:
                result["paper_categories"] = paper_categories
                
            return result
            
        except Exception as e:
            print(f"Error in process_query: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {
                "success": False,
                "response": f"I couldn't process your request due to an error: {str(e)}",
                "pending": False
            }
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt and return the response"""
        return await call_llm(prompt)
    
    def _extract_key_sections(self, paper_content: str) -> str:
        """Extract key sections from a paper to reduce token count"""
        # This is a simplified implementation
        # In a real system, you would use NLP to identify and extract key sections
        
        # Check if paper content is too large
        max_tokens = 8000  # Approximate token limit for model
        if len(paper_content) > max_tokens * 4:  # Rough estimate (4 chars per token)
            # Extract intro, methods, results, and conclusion sections
            sections = {
                "abstract": "",
                "introduction": "",
                "methods": "",
                "results": "",
                "discussion": "",
                "conclusion": ""
            }
            
            # Simple regex pattern matching for common section headers
            for section_name in sections.keys():
                pattern = rf"(?i)(^|\n)[\d\.\s]*({section_name}|{section_name.title()})[\s\:]*\n+(.*?)(?=\n[\d\.\s]*[A-Za-z\s]+[\s\:]*\n+|\Z)"
                match = re.search(pattern, paper_content, re.DOTALL)
                if match:
                    sections[section_name] = match.group(3).strip()
            
            # Combine the extracted sections
            extracted_content = ""
            for section_name, content in sections.items():
                if content:
                    extracted_content += f"\n\n## {section_name.title()}\n\n{content}"
            
            if extracted_content:
                return extracted_content
            
            # Fallback: If no sections were found, take beginning, middle, and end
            total_length = len(paper_content)
            beginning = paper_content[:2000]
            middle = paper_content[total_length//2-1000:total_length//2+1000]
            end = paper_content[-2000:]
            return f"{beginning}\n\n...[content truncated]...\n\n{middle}\n\n...[content truncated]...\n\n{end}"
        
        return paper_content
    
    async def _enrich_response(self, response: str, original_query: str) -> str:
        """Enrich the response with cross-references and additional context"""
        try:
            # Import here to avoid circular imports
            from src.services.vector_db import find_related_papers
            
            # Find related papers based on the query and response
            combined_text = f"{original_query} {response}"
            related_papers = await find_related_papers(combined_text, top_k=3)
            
            # Add related papers to the response if any were found
            if related_papers:
                related_content = "\n\n## Related Research\n\nYou might also be interested in these related papers:\n\n"
                for i, paper in enumerate(related_papers, 1):
                    title = paper.get("title", "Untitled")
                    authors = paper.get("authors", "Unknown authors")
                    year = paper.get("year", "")
                    url = paper.get("url", "")
                    
                    related_content += f"{i}. **{title}** ({year}). {authors}."
                    if url:
                        related_content += f" [Link]({url})"
                    related_content += "\n\n"
                
                # Append to the original response
                return f"{response}\n{related_content}"
            
            return response
        except Exception as e:
            print(f"Error enriching response: {str(e)}")
            # Return original response if there was an error
            return response
    
    async def categorize_paper(self, paper_content: str) -> List[str]:
        """
        Categorize a paper into relevant research areas
        This would use a fine-tuned model in a production system
        """
        # Simplified implementation - would use a fine-tuned model in production
        prompt = f"Categorize the following research paper into relevant academic categories:\n\n{paper_content[:1000]}..."
        
        response = await self._call_llm(prompt)
        
        # Extract categories from the response
        # This is a simple implementation - would use more robust parsing in production
        categories = [category.strip() for category in response.split(',')]
        return categories

# Create singleton instance
code_agent = CodeAgent()

async def process_user_query(prompt: str, paper_content: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a user query and return the response
    
    Args:
        prompt: The user's question
        paper_content: The content of the paper being analyzed
        
    Returns:
        Dict containing the response and any other relevant data
    """
    print(f"\n==== process_user_query called with prompt: {prompt[:50]}... ====")
    try:
        result = await code_agent.process_query(prompt, paper_content)
        print(f"process_user_query result: {result}")
        return result
    except Exception as e:
        print(f"Error in process_user_query: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {
            "success": False,
            "response": f"I couldn't process your request due to an error: {str(e)}",
            "pending": False
        } 