from googlenet_model import GoogLeNetSmall
import train_model as train
import torch.nn as nn
import torch.optim as optim
import torch


# Initialize model, criterion, and optimizer
model = GoogLeNetSmall()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train and save the model
print("Starting training...")
train.train_model(model, train.dataloader, criterion, optimizer)
torch.save(model.state_dict(), 'models/googlenet_model.pth')
print(f"Model training complete. Saved to models/googlenet_model.pth")