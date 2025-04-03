# Kaggle Research Papers Dataset

This directory contains the Kaggle research papers dataset used for the RAG system.

## Setup Instructions

1. Download the Kaggle dataset (papers.json) and place it in this directory
2. Run the load_kaggle_papers.py script to process the papers:
   ```bash
   python load_kaggle_papers.py
   ```
3. The script will:
   - Load the papers from papers.json
   - Process each paper to extract content and metadata
   - Generate embeddings and categories
   - Store everything in the vector database
   - Save the embeddings to kaggle_embeddings.pkl

## Dataset Format

The papers.json file should contain an array of paper objects with the following structure:
```json
{
  "paper_id": "unique_id",
  "title": "Paper Title",
  "authors": "Author Names",
  "year": "Publication Year",
  "abstract": "Paper Abstract",
  "body_text": "Full Paper Text",
  "url": "Paper URL"
}
```

## Notes

- The embeddings are saved to kaggle_embeddings.pkl and will be automatically loaded when the server starts
- The vector database uses these embeddings to find related papers and provide better context for queries
- You can update the dataset by replacing papers.json and rerunning the load script 