# Training and Using Custom Models with RAG

This guide explains how to train a custom model for paper categorization and integrate it with the existing Retrieval-Augmented Generation (RAG) implementation.

## Overview

The system has two main components:

1. **RAG Implementation**: Uses the existing Llama 3 model through the custom API endpoint to generate responses.
2. **Paper Categorization**: Can use either:
   - Rule-based categorization (default)
   - A trained machine learning model (after following this guide)

## Key Features

- **Uses Actual Dataset Categories**: The model learns from and predicts the actual categories present in your dataset (like cs.LG, cs.AI, cs.CV, etc.) instead of using predefined labels.
- **Multi-label Classification**: Papers can belong to multiple categories simultaneously.
- **Seamless Integration**: The trained model automatically works with your existing RAG implementation.
- **Fallback Mechanism**: If the model has issues, the system falls back to rule-based categorization.

## Prerequisites

- Python 3.9+
- Virtual environment (recommended)
- GPU with CUDA support (recommended, but not required)
- The CS papers dataset (`server/data/cs_papers_api.csv`)

## Setup and Training

### Step 1: Activate Your Virtual Environment

```bash
cd server
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### Step 2: Install CUDA-enabled PyTorch (if using GPU)

If you have a CUDA-compatible GPU, install the CUDA version of PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Run the Setup and Training Script

Our setup script handles dependency installation and initiates the training process:

```bash
python scripts/setup_training.py
```

This will:
1. Install all required dependencies
2. Check for GPU availability
3. Process the CS papers dataset
4. Extract the actual categories from the dataset
5. Train the model on these categories
6. Save the trained model and category mapping

### Step 4: Customizing Training Parameters

You can customize training parameters:

```bash
python scripts/setup_training.py --model_name distilbert-base-uncased --max_samples 5000 --epochs 5
```

Available parameters:
- `--csv_path`: Path to the CSV file (default: `./data/cs_papers_api.csv`)
- `--model_name`: Base model name (default: `distilbert-base-uncased`)
- `--output_dir`: Directory to save the model (default: `./data/trained_model`)
- `--max_samples`: Maximum number of samples to use (default: `10000`)
- `--batch_size`: Batch size (default: `4`)
- `--epochs`: Number of epochs (default: `3`)
- `--learning_rate`: Learning rate (default: `2e-5`)
- `--skip_setup`: Skip dependency installation

### Step 5: Restart the Server

After training, restart the server to load the new model:

```bash
python app.py
```

The system will automatically detect and load the trained model. If there's any issue with the model, it will fall back to the rule-based categorization method.

## Advanced Usage

### Running Training Directly

If you've already set up the environment, you can run the training script directly:

```bash
python scripts/train_model.py --max_samples 20000 --epochs 5
```

### Choosing Different Base Models

You can experiment with different pre-trained models:

```bash
python scripts/setup_training.py --model_name bert-base-uncased
```

Some recommended models:
- `distilbert-base-uncased`: Small, fast, good for CPU training
- `bert-base-uncased`: Good balance of accuracy and speed
- `roberta-base`: Higher accuracy, more resource-intensive

### Training with Limited Resources

If you have limited computational resources:

```bash
python scripts/setup_training.py --max_samples 1000 --batch_size 2 --epochs 2
```

## How It Works

1. **Dataset Processing**:
   - The script reads your CSV file and extracts all unique categories (like cs.LG, cs.AI, etc.)
   - It preserves the original category names from your dataset
   - It maintains the natural distribution of categories

2. **Model Training**:
   - The script tokenizes paper texts (title + abstract)
   - It converts the categories to one-hot encoding
   - It trains a multi-label classification model to predict the actual dataset categories

3. **Integration with RAG**:
   - The trained model is automatically detected and loaded by `model_service.py`
   - `llm_service.py` uses the model service for paper categorization
   - The RAG implementation uses these categories to enhance retrieval

4. **Fallback Mechanism**:
   - If the model fails or isn't available, the system falls back to rule-based categorization
   - This ensures the system always works, even without a trained model

## Troubleshooting

### CUDA Issues

If you encounter CUDA errors:

```bash
python scripts/setup_training.py --model_name distilbert-base-uncased --batch_size 1
```

Or train on CPU:

```bash
# Force CPU training by setting CUDA_VISIBLE_DEVICES
set CUDA_VISIBLE_DEVICES=-1  # Windows
export CUDA_VISIBLE_DEVICES=-1  # Linux/Mac
python scripts/setup_training.py
```

### Out of Memory Errors

If you encounter out of memory errors:

1. Reduce batch size: `--batch_size 1`
2. Reduce maximum samples: `--max_samples 1000`
3. Use a smaller model: `--model_name distilbert-base-uncased`

### Dependency Conflicts

If you encounter dependency conflicts:

```bash
pip install numpy>=1.21.6,<1.28.0
pip install -r requirements.txt
```

## Next Steps

- Experiment with different models and hyperparameters
- Collect user feedback on categorization accuracy
- Consider fine-tuning the Llama model for specific domains 