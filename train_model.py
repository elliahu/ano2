import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Paths to training data
TRAIN_PATH = "train_images"

# Define transformations for the training images
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Load dataset
dataset = ImageFolder(root=TRAIN_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            if model.training:
                # Unpack the outputs
                main_output, aux1_output, aux2_output = outputs
                loss_main = criterion(main_output, labels)
                loss_aux1 = criterion(aux1_output, labels)
                loss_aux2 = criterion(aux2_output, labels)
                
                # Combine losses with auxiliary weights
                loss = loss_main + 0.3 * loss_aux1 + 0.3 * loss_aux2
            else:
                # Only use main output if model is in eval mode
                loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
