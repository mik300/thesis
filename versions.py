import torch
import torchvision
import sys

# Print Python version
print(f"Python version: {sys.version}")

# Print PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Print TorchVision version
print(f"TorchVision version: {torchvision.__version__}")

print(f"Cuda version for PyTorch: {torch.version.cuda}")

print(f'Cuda is available: {torch.cuda.is_available()}\n')

