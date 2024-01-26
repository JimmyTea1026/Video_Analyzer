import torch

try:
    # Check if PyTorch is available
    torch_version = torch.__version__
    print(f"PyTorch version {torch_version} is available.")
except ImportError:
    print("PyTorch is not installed. Please install it using 'pip install torch'.")
except Exception as e:
    print(f"An error occurred while checking PyTorch availability: {e}")
