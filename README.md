# Research Paper Assistant

A web application that helps users analyze and understand research papers. This application uses natural language processing and RAG (Retrieval-Augmented Generation) to provide enhanced paper analysis.

## Features

- PDF processing and text extraction
- Research paper analysis with LLM
- Paper categorization based on content
- Cross-referencing with related research (RAG)
- Interactive chat interface

## Architecture

### Client
- React frontend with TypeScript and Tailwind CSS
- PDF viewer and text selection
- Chat interface for interacting with the assistant
- Real-time response updates

### Server
- Flask backend with SQLite for data storage
- LLM API integration for language processing
- PDF processing with pdfplumber
- Vector database for document retrieval and RAG
- Async processing for long-running queries

## Installation

### Prerequisites
- Python 3.9 or higher
- Node.js 16 or higher
- LLM API access (OpenAI or compatible endpoint)

### Server Setup
1. Clone the repository
2. Navigate to the server directory: `cd server`
3. Create a virtual environment: `python -m venv venv`
4. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
5. Install dependencies: `pip install -r requirements.txt`
6. Copy `.env.example` to `.env` and update with your credentials:
   ```
   SQLALCHEMY_DATABASE_URI_DEV=sqlite:///db.sqlite3
   SQLALCHEMY_TRACK_MODIFICATIONS=False
   SECRET_KEY=your_secret_key
   OPENAI_API_KEY=your_api_key
   OPENAI_API_BASE=https://api.openai.com/v1  # or your custom endpoint
   OPENAI_MODEL=llama3p1-8b-instruct  # or your preferred model
   EMBEDDING_MODEL=text-embedding-ada-002  # or your preferred embedding model
   ```
7. Initialize the database: `python app.py` (this will create the database if it doesn't exist)
8. Run the server: `python app.py`

### Client Setup
1. Navigate to the client directory: `cd client`
2. Install dependencies: `npm install`
3. Copy `.env.example` to `.env` and update:
   ```
   VITE_SERVER_URL=http://localhost:8000
   ```
4. Run the client: `npm run dev`
5. Access the application at http://localhost:5173

## Usage

1. Open the application in your web browser
2. Upload a PDF or paste text from a research paper
3. Ask questions about the paper in the chat interface
4. The assistant will analyze the paper and provide detailed responses, including cross-references to related research

## Data Processing

The application includes scripts for processing research papers:

1. Load Kaggle papers:
   ```
   python server/load_kaggle_papers.py
   ```
   This will process the papers and generate embeddings for the RAG system.

2. Clear queries:
   You can clear the query history through the API:
   ```
   curl -X DELETE http://localhost:8000/api/queries/
   ```
   Or use the provided HTML interface at `client/public/clear-queries.html`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Inspired by OpenAI's Deep Research
- Thanks to Hugging Face's Open Deep Research blog post 