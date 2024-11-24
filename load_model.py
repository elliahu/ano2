import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from parking_model import _train
import os



def load_pytorch_model(model, model_path='models/model.pth'):
    if not os.path.exists(model_path):
        _train(model, nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001), model_path)
    print(f"Model loaded from {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model