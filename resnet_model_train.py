from resnet_model import ResNet18
import train_model as train
import torch.nn as nn
import torch.optim as optim
import torch

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Since ResNet only returns the main output
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Initialize model, criterion, and optimizer
model = ResNet18()
num_epochs = 15
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train and save the model
print("Starting training...")
train_model(model, train.dataloader, criterion, optimizer, num_epochs)
torch.save(model.state_dict(), 'models/resnet18_model.pth')
print(f"Model training complete. Saved to models/resnet18_model.pth")