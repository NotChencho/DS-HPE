import sys
import torch
print("venv python:", sys.executable)
print("torch.__version__:", getattr(torch, "__version__", None))
print("torch.version.cuda:", getattr(getattr(torch, 'version', None), 'cuda', None))
print("torch.cuda.is_available():", torch.cuda.is_available())
