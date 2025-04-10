# CS4973 - Engineering LLM-Integrated Systems Final Project

# Research Paper Assistant

A full-stack application that helps users analyze and understand computer science research papers using LLM-powered Retrieval-Augmented Generation (RAG).

## Features

- PDF extraction and processing of research papers
- Vector embeddings and semantic search for relevant content retrieval
- Automatic paper categorization using both rule-based and ML approaches
- Context-enhanced LLM responses for technical questions about paper content
- Real-time responses via WebSockets
- Paper similarity search to find related research

## Project Structure

```
cs4973-final-project/
├── client/             # React+TypeScript frontend
├── server/             # Python/Flask backend
│   ├── data/           # Dataset storage
│   ├── src/            # Core server code
│   │   ├── config/     # Configuration files
│   │   ├── controllers/# API controllers
│   │   ├── models/     # Database models
│   │   ├── services/   # Business logic services
│   ├── scripts/        # Utility scripts
│   ├── TRAINING.md     # Model training documentation
```

## Prerequisites

- Python 3.9 or higher
- Node.js 18 or higher
- pnpm (for client dependency management)

## Setup Instructions

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd cs4973-final-project
```

### Step 2: Set Up the Server

```bash
cd server

# Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Copy the template and edit with your credentials
cp .env.example .env
```

Edit the `.env` file with your configuration:
```
SQLALCHEMY_DATABASE_URI_DEV=sqlite:///db.sqlite3
SQLALCHEMY_TRACK_MODIFICATIONS=False
SECRET_KEY=your_secret_key
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://api.openai.com/v1  # or your custom endpoint
OPENAI_MODEL=llama3p1-8b-instruct  # or your preferred model
EMBEDDING_MODEL=text-embedding-ada-002  # or your preferred embedding model
```

### Step 3: Set Up the Client

```bash
cd ../client

# Install dependencies using pnpm
pnpm install

# Set up environment variables
# Create a .env file with the server URL
echo "SERVER_URL=http://localhost:8000" > .env
```

### Step 4: Load Paper Dataset

The system requires a dataset of computer science papers. You can use the included script to load papers from a CSV file:

1. Place your CS papers CSV dataset in the `server/data/` directory with filename `cs_papers_api.csv`.
   The CSV should have columns: paper_id, title, abstract, categories, year.

2. Run the loading script:

```bash
cd server
python load_kaggle_papers.py
```

This will:
- Process the papers from the CSV
- Generate embeddings for each paper
- Save the embeddings to `cs_papers_embeddings.pkl`

### Step 5: Initialize Database

```bash
cd server
# Set up the database
flask db init
flask db migrate -m "Initial migration"
flask db upgrade
```

## Running the Application

### Start the Server

```bash
cd server
python app.py
```

The server will start at http://localhost:8000

### Start the Client

```bash
cd client
npm run dev
```

The client will start at http://localhost:5173

## Using the Application

1. **Upload a Paper**: Use the PDF upload functionality to analyze a research paper.

2. **Ask Questions**: Enter queries about the paper content. The system will:
   - Process your query
   - Retrieve relevant information from the paper
   - Augment the LLM's response with specific context
   - Categorize the paper based on its content
   - Provide related papers from the database

3. **Browse History**: View your past queries and responses.

## Advanced Features

### Training a Custom Categorization Model

You can train a custom model for paper categorization. See `server/TRAINING.md` for detailed instructions.

Basic training command:
```bash
cd server
python scripts/setup_training.py
```

### Using a Different LLM Provider

To use a different LLM provider:
1. Update the `OPENAI_API_BASE` and `OPENAI_API_KEY` in your `.env` file
2. Make sure the `OPENAI_MODEL` is compatible with your provider

## Troubleshooting

### Missing Embeddings File

If you see a warning about `cs_papers_embeddings.pkl` not found, you need to load the paper dataset as described in Step 4.

### OpenAI API Issues

If you encounter OpenAI API issues, check:
- Your API key is valid
- Your API base URL is correct
- You have sufficient quota/credits

### Database Errors

If you encounter database errors:
```bash
cd server
# Remove the existing database and migrations
rm -rf instance/db.sqlite3 migrations/

# Reinitialize the database
flask db init
flask db migrate -m "Initial migration"
flask db upgrade
```

## Further Development

- Implement custom LLM fine-tuning for specific paper domains
- Add visualization tools for paper relationships
- Enhance PDF parsing capabilities for complex layouts
- Integrate with academic search APIs for expanded paper discovery

## License

This project is licensed under the MIT License - see the LICENSE file for details.
