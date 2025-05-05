import torch

print(f"PyTorch version: {torch.__version__}")

# Check for CUDA (NVIDIA GPU)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA devices count: {torch.cuda.device_count()}")

# Check for MPS (Apple Silicon GPU - M1/M2/M3)
try:
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        print("MPS (Apple Silicon GPU) is available")
        mps_device = torch.device("mps")
        # Create a simple tensor and move it to MPS device to verify it works
        x = torch.ones(1)
        x = x.to(mps_device)
        print(f"Successfully created tensor on MPS device: {x.device}")
    else:
        if hasattr(torch, 'mps'):
            print("MPS is supported in this PyTorch build, but MPS device is not available")
            print(f"MPS device available: {torch.backends.mps.is_available()}")
            print(f"MPS backend available: {torch.backends.mps.is_built()}")
        else:
            print("MPS is not supported in this PyTorch build")
except Exception as e:
    print(f"Error checking MPS availability: {str(e)}")

# Summary
if torch.cuda.is_available():
    print("\nYou can use NVIDIA GPU for training")
elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
    print("\nYou can use Apple Silicon GPU for training")
else:
    print("\nNo GPU available. Training will run on CPU (which will be much slower)")