#!/usr/bin/env python3
"""
Script to set up the training environment and run the training process
"""

import os
import sys
import subprocess
import argparse
import platform

# Check if running in virtual environment
def in_virtualenv():
    """Check if running in a virtual environment"""
    return (hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))

def install_dependencies():
    """Install required dependencies for training"""
    print("Installing required dependencies...")
    
    # Basic dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "datasets", "scikit-learn", "pandas", "numpy"])
    
    # Try installing specific version requirements
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy>=1.21.6,<1.28.0"])
    except:
        print("Warning: Could not install specific numpy version range. Using default version.")
    
    print("Dependencies installed successfully!")
    
    # Verify PyTorch CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
            print(f"\nPyTorch successfully installed with CUDA support!")
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA version: {cuda_version}")
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("\nPyTorch installed but CUDA not available. Training will use CPU (slower).")
            print("If you have a CUDA-capable GPU, try reinstalling PyTorch with:")
            print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    except ImportError:
        print("\nFailed to import torch. Please reinstall PyTorch.")

def check_gpu():
    """Check for GPU availability"""
    print("Checking for GPU availability...")
    
    try:
        import torch
        
        # Print detailed information about CUDA
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'Unknown'}")
            print(f"Device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("No GPU detected. Training will run on CPU (much slower).")
            print("\nTroubleshooting tips:")
            print("1. Make sure you have a CUDA-compatible GPU")
            print("2. Make sure you have installed the CUDA toolkit")
            print("3. Make sure you've installed the CUDA version of PyTorch with:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            print("4. Check NVIDIA drivers are up to date")
            
            # Try to detect if NVIDIA drivers are installed
            try:
                result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode == 0:
                    print("\nNVIDIA drivers are installed, but PyTorch can't detect CUDA.")
                    print("Output of nvidia-smi:")
                    print(result.stdout)
                else:
                    print("\nFailed to run nvidia-smi. NVIDIA drivers may not be installed.")
            except FileNotFoundError:
                print("\nNVIDIA driver utilities (nvidia-smi) not found. Please install NVIDIA drivers.")
            
            return False
    except:
        print("Error checking GPU. Make sure PyTorch is installed correctly.")
        return False

def run_training(args):
    """Run the model training"""
    print("Starting model training...")
    
    # Construct command with provided arguments
    command = [
        sys.executable, "scripts/train_model.py",
        "--csv_path", args.csv_path,
        "--model_name", args.model_name,
        "--output_dir", args.output_dir,
        "--max_samples", str(args.max_samples),
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--learning_rate", str(args.learning_rate)
    ]
    
    # Run the training script
    try:
        subprocess.check_call(command)
        print("Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running training: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Set up training environment and run training')
    parser.add_argument('--csv_path', default='./data/cs_papers_api.csv', help='Path to the CSV file')
    parser.add_argument('--model_name', default='distilbert-base-uncased', help='Base model name')
    parser.add_argument('--output_dir', default='./data/trained_model', help='Directory to save the model')
    parser.add_argument('--max_samples', type=int, default=10000, help='Maximum number of samples to use')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--skip_setup', action='store_true', help='Skip dependency installation')
    
    args = parser.parse_args()
    
    # Check if running in virtual environment
    if not in_virtualenv():
        print("Warning: It's recommended to run this script in a virtual environment.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Install dependencies if not skipped
    if not args.skip_setup:
        install_dependencies()
    
    # Check for GPU
    has_gpu = check_gpu()
    
    # Adjust batch size for CPU training
    if not has_gpu and args.batch_size > 2:
        print("Reducing batch size to 2 for CPU training...")
        args.batch_size = 2
    
    # Run the training
    success = run_training(args)
    
    if success:
        # Print instructions
        print("\nYour model has been trained and saved successfully!")
        print(f"Model directory: {os.path.abspath(args.output_dir)}")
        print("\nTo use your model in the application:")
        print("1. Restart the server")
        print("2. The model will be automatically loaded and used for paper categorization")
    
if __name__ == '__main__':
    main() 