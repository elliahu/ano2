import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import train_model as train

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

def _train():
    # Initialize model, criterion, and optimizer
    model = ParkingSpaceClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and save the model
    print("Starting training...")
    train.train_model(model, train.dataloader, criterion, optimizer)
    torch.save(model.state_dict(), 'model/parking_space_cnn.pth')
    print(f"Model training complete. Saved to model/parking_space_cnn.pth")