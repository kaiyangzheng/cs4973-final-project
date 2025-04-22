import os
import asyncio
from typing import Dict, List, Optional, Any
from .pdf_service import extract_text_from_pdf_base64, is_pdf_content, extract_metadata_from_pdf
import re
from .rag_service import rag_service
from .llm_client import call_llm
from src.models.query_model import UserQuery
from src import db
from .model_service import model_service  # Import the new model service

class CodeAgent:
    """
    A code agent that uses LLM to reason about and understand research papers.
    It implements deep research capabilities by categorizing papers and performing
    cross-referencing to provide better context.
    """
    
    def __init__(self):
        self.paper_categories = []
        self.context = {}
        # Store conversation history by socket_id
        self.conversation_history = {}
    
    async def process_query(self, prompt: str, paper_content: Optional[str] = None, socket_id: Optional[str] = None, use_rag: bool = True) -> Dict[str, Any]:
        """
        Process a user query about a research paper
        
        Args:
            prompt: The user's question
            paper_content: The content of the paper being analyzed
            socket_id: The socket ID of the client, used for maintaining conversation context
            
        Returns:
            Dict containing the response and any other relevant data
        """
        print(f"\n==== process_query called with prompt: {prompt[:50]}... ====")
        try:
            # Get previous conversation history for this socket
            conversation_context = []
            if socket_id:
                # Retrieve previous messages for this socket_id
                previous_messages = await self._get_conversation_history(socket_id)
                if previous_messages:
                    conversation_context = previous_messages
                    print(f"Retrieved {len(previous_messages)} previous messages for socket {socket_id}")
            
            # Process paper content if provided
            paper_text = paper_content
            paper_metadata = None
            paper_categories = []
            
            if paper_text:
                print(f"Processing paper content of length: {len(paper_text)}")
                
                # Categorize the paper using the new model service
                try:
                    from src.services.vector_db import vector_db
                    
                    # Generate a unique ID based on content hash
                    import hashlib
                    paper_id = hashlib.md5(paper_text[:5000].encode('utf-8')).hexdigest()
                    
                    # Categorize the paper using the model service
                    print("Categorizing paper using model service...")
                    paper_categories = model_service.predict_categories(paper_text)
                    print(f"Paper categories: {paper_categories}")  
                    
                    # Add categories to vector database
                    vector_db.add_paper_category(paper_id, paper_categories)
                    
                except Exception as e:
                    print(f"Error categorizing paper: {str(e)}")
                    paper_categories = []
            
            # Prepare the context with conversation history
            enhanced_prompt = self._prepare_enhanced_prompt(prompt, conversation_context)
            
            # Generate response using RAG
            print("Generating response using RAG...")
            # We'll let the RAG service handle the paper categories internally
            response = await rag_service.generate_response(enhanced_prompt, paper_text, use_rag)
            print(f"RAG response received, length: {len(response)}")
            
            # Store this interaction in conversation history if socket_id is provided
            if socket_id:
                # Also store paper categories if available
                self._update_conversation_history(socket_id, prompt, response)
            
            # Prepare the final response
            result = {
                "success": True,
                "response": response,
                "pending": False
            }
            
            # Add paper categories to the response if available
            if paper_categories:
                try:
                    # Get human-readable labels for the categories
                    category_labels = [
                        {"code": cat, "label": model_service.get_category_label(cat)}
                        for cat in paper_categories
                    ]
                    # Include under both keys for compatibility
                    result["categories"] = category_labels
                    result["paper_categories"] = category_labels
                except Exception as e:
                    print(f"Error adding category labels to response: {str(e)}")
                    # Fallback to just using the codes
                    simple_categories = [{"code": cat, "label": cat} for cat in paper_categories]
                    result["categories"] = simple_categories
                    result["paper_categories"] = simple_categories
            
            return result
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "response": f"Sorry, I encountered an error while processing your query. Please try again later. Error: {str(e)}",
                "pending": False
            }
    
    async def _get_conversation_history(self, socket_id: str) -> List[Dict[str, str]]:
        """
        Retrieve conversation history for the given socket_id from the database
        
        Args:
            socket_id: The socket ID to retrieve history for
            
        Returns:
            List of conversation messages in the format [{"role": "user", "content": "..."},
                                                        {"role": "assistant", "content": "..."}]
        """
        try:
            # Get previous queries for this socket_id, ordered by creation time
            previous_queries = UserQuery.query.filter_by(socket_id=socket_id) \
                                        .filter_by(pending=False) \
                                        .order_by(UserQuery.created_at.asc()) \
                                        .all()
            
            # Format the queries into a conversation format
            conversation = []
            for query in previous_queries:
                conversation.append({"role": "user", "content": query.prompt})
                if query.response:
                    conversation.append({"role": "assistant", "content": query.response})
            
            # Only keep the last 5 interactions (10 messages) to avoid context overflow
            if len(conversation) > 10:
                conversation = conversation[-10:]
                
            return conversation
        except Exception as e:
            print(f"Error retrieving conversation history: {str(e)}")
            return []
    
    def _update_conversation_history(self, socket_id: str, prompt: str, response: str) -> None:
        """
        Update in-memory conversation history with the latest interaction
        
        Args:
            socket_id: The socket ID to update history for
            prompt: The user's prompt
            response: The assistant's response
        """
        if socket_id not in self.conversation_history:
            self.conversation_history[socket_id] = []
            
        # Add the new messages
        self.conversation_history[socket_id].append({"role": "user", "content": prompt})
        self.conversation_history[socket_id].append({"role": "assistant", "content": response})
        
        # Limit the history size (keep last 5 interactions = 10 messages)
        if len(self.conversation_history[socket_id]) > 10:
            self.conversation_history[socket_id] = self.conversation_history[socket_id][-10:]
            
        print(f"Updated conversation history for socket {socket_id}, now has {len(self.conversation_history[socket_id])} messages")
    
    def _prepare_enhanced_prompt(self, original_prompt: str, conversation_history: List[Dict[str, str]]) -> str:
        """
        Prepare an enhanced prompt that includes conversation history
        
        Args:
            original_prompt: The user's original prompt
            conversation_history: List of previous conversation messages
            
        Returns:
            Enhanced prompt with conversation context
        """
        if not conversation_history:
            return original_prompt
            
        # Construct a context string from conversation history
        context_string = "Previous conversation:\n"
        for message in conversation_history:
            role = "User" if message["role"] == "user" else "Assistant"
            context_string += f"{role}: {message['content']}\n\n"
            
        # Format the final prompt
        enhanced_prompt = f"""
This is a conversation with a user about a research paper. Here is the conversation history:

{context_string}

Current user question: {original_prompt}

Based on the conversation history and the current question, provide a relevant and helpful response.
"""
        
        return enhanced_prompt
    
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

async def process_user_query(prompt: str, paper_content: Optional[str] = None, socket_id: Optional[str] = None, use_rag: bool = True) -> Dict[str, Any]:
    """
    Process a user query and return the response
    
    Args:
        prompt: The user's question
        paper_content: The content of the paper being analyzed
        socket_id: The socket ID for maintaining conversation context
        
    Returns:
        Dict containing the response and any other relevant data
    """
    print(f"\n==== process_user_query called with prompt: {prompt[:50]}... ====")
    try:
        result = await code_agent.process_query(prompt, paper_content, socket_id, use_rag)
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