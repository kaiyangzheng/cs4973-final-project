# Research Paper Assistant - Server

The server-side application for the Research Paper Assistant, built with Flask and Python.

## Features

- PDF text extraction and processing
- LLM integration for paper analysis
- Vector database for document retrieval
- Paper categorization
- Async query processing
- SQLite database for query storage

## Project Structure

```
server/
├── src/
│   ├── services/       # Core services
│   │   ├── llm_client.py    # LLM API client
│   │   ├── pdf_service.py   # PDF processing
│   │   └── vector_db.py     # Vector database
│   ├── models/         # Database models
│   └── routes/         # API routes
├── scripts/            # Utility scripts
├── data/              # Data storage
├── .env.example       # Environment variables template
└── requirements.txt   # Python dependencies
```

## Development

### Prerequisites
- Python 3.9 or higher
- Virtual environment (recommended)

### Setup
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file from the template:
   ```bash
   cp .env.example .env
   ```

4. Update the `.env` file with your credentials:
   ```
   SQLALCHEMY_DATABASE_URI_DEV=sqlite:///db.sqlite3
   SQLALCHEMY_TRACK_MODIFICATIONS=False
   SECRET_KEY=your_secret_key
   OPENAI_API_KEY=your_api_key
   OPENAI_API_BASE=https://api.openai.com/v1  # or your custom endpoint
   OPENAI_MODEL=llama3p1-8b-instruct  # or your preferred model
   EMBEDDING_MODEL=text-embedding-ada-002  # or your preferred embedding model
   ```

### Running the Server
```bash
python app.py
```

The server will be available at http://localhost:8000

## API Endpoints

### Queries
- `POST /api/queries/` - Submit a new query
  - Request body: `{ "prompt": string, "paper_content": string }`
  - Response: Query object with status and response

- `GET /api/queries/` - Get query history
  - Response: Array of query objects

- `DELETE /api/queries/` - Clear query history
  - Response: Success message

## Data Processing

### Loading Kaggle Papers
The server includes a script to process and load research papers from Kaggle:

```bash
python load_kaggle_papers.py
```

This script will:
1. Process the papers
2. Generate embeddings
3. Store them in the vector database

## Contributing

Please follow the project's coding standards and submit pull requests for any improvements.


