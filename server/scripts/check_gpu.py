#!/usr/bin/env python3
"""
Script to check GPU availability and CUDA setup
"""

import os
import sys
import subprocess
import platform

def main():
    """Main function to check GPU and CUDA status"""
    print("===== GPU and CUDA Diagnostic Tool =====")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print("=" * 40)
    
    # Check PyTorch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available in PyTorch: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device count: {torch.cuda.device_count()}")
            print(f"Current GPU device: {torch.cuda.current_device()}")
            print(f"GPU device name: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
            
            # Create a small tensor on GPU to test functionality
            print("\nTesting GPU tensor creation...")
            try:
                x = torch.rand(1000, 1000).cuda()
                print(f"Successfully created tensor on GPU with shape {x.shape}")
                del x
                torch.cuda.empty_cache()
                print("GPU test successful!")
            except Exception as e:
                print(f"Error creating tensor on GPU: {e}")
        else:
            print("\nCUDA is not available. Checking system...")
    except ImportError:
        print("PyTorch is not installed. Please install it with:")
        print("pip install torch torchvision torchaudio")
        return
    
    # Try checking nvidia-smi
    print("\n--- Checking NVIDIA System Management Interface ---")
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("nvidia-smi output:")
            print(result.stdout)
        else:
            print("Failed to run nvidia-smi. Error:")
            print(result.stderr)
    except FileNotFoundError:
        print("nvidia-smi not found. NVIDIA drivers may not be installed or in PATH.")
    
    # Check environment variables
    print("\n--- Checking Environment Variables ---")
    for var in ['CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES']:
        print(f"{var}: {os.environ.get(var, 'Not set')}")
    
    # Check paths
    print("\n--- Checking System PATH for CUDA ---")
    paths = os.environ.get('PATH', '').split(os.pathsep)
    cuda_paths = [p for p in paths if 'cuda' in p.lower()]
    if cuda_paths:
        print("CUDA in PATH:")
        for path in cuda_paths:
            print(f"  - {path}")
    else:
        print("No CUDA directories found in PATH")
    
    # Windows-specific registry check
    if platform.system() == 'Windows':
        print("\n--- Checking Windows Registry for NVIDIA ---")
        try:
            import winreg
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\NVIDIA Corporation\Global") as key:
                    print("NVIDIA registry entries found")
            except WindowsError:
                print("NVIDIA registry entries not found")
        except ImportError:
            print("Could not check registry (winreg not available)")
    
    # Troubleshooting tips
    print("\n===== Troubleshooting Tips =====")
    if not torch.cuda.is_available():
        print("1. Make sure you have installed NVIDIA GPU drivers")
        print("2. Make sure you have installed the CUDA Toolkit")
        print("3. Make sure you've installed the CUDA version of PyTorch:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("4. Check that your GPU is CUDA-compatible")
        print("5. On Windows, verify that Developer Mode is enabled or you're running as administrator")
    
    print("\n===== Diagnostic Complete =====")

if __name__ == "__main__":
    main() 