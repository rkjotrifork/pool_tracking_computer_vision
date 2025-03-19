import torch
from torchvision import models

print("PyTorch Version:", torch.__version__)
print(torch.backends.mps.is_available())  # Should return True
print(torch.backends.mps.is_built())  # Should return True

device = torch.device("mps")

print(device)