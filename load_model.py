import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

def load_pytorch_model(model, model_path='models/model.pth'):
    print(f"Model loaded from {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model