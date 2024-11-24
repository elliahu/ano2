import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import ToTensor

class ParkingSpaceClassifier(nn.Module):
    def __init__(self):
        super(ParkingSpaceClassifier, self).__init__()
        # Define a small CNN for binary classification (occupied or empty)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # Adjust based on input image size
        self.fc2 = nn.Linear(128, 2)  # Output layer for binary classification

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ParkingSpaceModelS(nn.Module):
    def __init__(self):
        super(ParkingSpaceModelS, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ParkingSpaceModelM(nn.Module):
    def __init__(self):
        super(ParkingSpaceModelM, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def classify_image_with_pytorch(model, image):
    # Preprocess the image for the model
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item() == 1  # Returns True if occupied

def _train(model, criterion, optimizer, model_path, epochs=10, batch_size=32):
    # Prepare the dataset
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Ensure all images are 32x32
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(
        root="train_images",
        transform=transform
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    print("Starting training...")

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Log the epoch loss
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model training complete. Saved to {model_path}")