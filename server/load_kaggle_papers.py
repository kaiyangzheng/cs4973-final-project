import os
import json
import asyncio
import time
from src.services.vector_db import vector_db, _get_hash_embedding
from src.services.llm_client import call_llm

# Configuration
USE_OPENAI_EMBEDDINGS = False  # Set to True to use OpenAI embeddings (requires API quota)
BATCH_SIZE = 10000  # Process papers in batches
DELAY_BETWEEN_BATCHES = 0  # Seconds to wait between batches

async def load_kaggle_papers(csv_path: str):
    """
    Load and process papers from the CS papers API CSV
    
    Args:
        csv_path: Path to the CSV file containing paper data
    """
    try:
        # Load papers from CSV
        import csv
        papers = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            papers = list(reader)
        
        total_papers = len(papers)
        print(f"Loaded {total_papers} papers from {csv_path}")
        print(f"Using {'OpenAI' if USE_OPENAI_EMBEDDINGS else 'hash-based'} embeddings")
        
        # Process papers in batches
        for batch_start in range(0, total_papers, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_papers)
            batch = papers[batch_start:batch_end]
            
            for i, paper in enumerate(batch, batch_start + 1):
                try:
                    # Extract content
                    content = paper.get('abstract', '')
                    if not content:
                        continue
                    
                    # Generate unique ID
                    paper_id = paper.get('paper_id', '')
                    
                    # Get categories
                    categories = paper.get('categories', '').split(',')
                    if not categories or not categories[0]:
                        # Use LLM to categorize if no categories found
                        prompt = f"Categorize the following research paper abstract into relevant academic categories:\n\n{content[:1000]}"
                        response = await call_llm(prompt)
                        categories = [cat.strip() for cat in response.split(',')]
                    
                    # Create metadata
                    metadata = {
                        "id": paper_id,
                        "title": paper.get('title', 'Untitled'),
                        "authors": "Unknown",  # CSV doesn't have authors
                        "year": paper.get('year', ''),
                        "categories": categories,
                        "abstract": content,
                        "introduction": "",  # These would be extracted in a real system
                        "conclusion": "",
                        "url": f"https://arxiv.org/abs/{paper_id}" if paper_id else ""
                    }
                    
                    # Generate embedding
                    if USE_OPENAI_EMBEDDINGS:
                        try:
                            from src.services.vector_db import get_embedding
                            embedding = await get_embedding(content)
                        except Exception as e:
                            print(f"Error getting OpenAI embedding for paper {i}/{total_papers}: {str(e)}")
                            print("Falling back to hash-based embedding")
                            embedding = _get_hash_embedding(content)
                    else:
                        embedding = _get_hash_embedding(content)
                    
                    # Add to vector database
                    vector_db.add_paper_category(paper_id, categories)
                    vector_db.add_vector(embedding, metadata)
                    
                    # Print progress
                    print(f"Processed paper {i}/{total_papers}: {metadata['title']}")
                    
                except Exception as e:
                    print(f"Error processing paper {i}/{total_papers}: {str(e)}")
                    continue
            
            # Save progress after each batch
            vector_db.save_embeddings('cs_papers_embeddings.pkl')
            print(f"Saved progress after batch {batch_start//BATCH_SIZE + 1}")
            
            # Wait between batches to avoid rate limits
            if batch_end < total_papers:
                print(f"Waiting {DELAY_BETWEEN_BATCHES} seconds before next batch...")
                time.sleep(DELAY_BETWEEN_BATCHES)
        
        print("Finished processing all papers")
        print("Saved final embeddings to cs_papers_embeddings.pkl")
        
    except Exception as e:
        print(f"Error loading papers: {str(e)}")
        raise

if __name__ == "__main__":
    # Construct path to CSV file
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'cs_papers_api.csv')
    
    # Run the async function
    asyncio.run(load_kaggle_papers(csv_path)) 