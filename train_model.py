import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from augmented_dataset import AugmentedDataset
from gan import Generator, Discriminator

# Paths to training data
TRAIN_PATH = "train_images"
MODEL_PATH = "models/gan_model.pth"

# Define transformations for the training images
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Initialize the generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Load the pre-trained GAN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = generator.to(device)
discriminator = discriminator.to(device)

# Function to load the model
def load_gan(generator, discriminator, optimizer_g, optimizer_d, filename="models/gan_model.pth"):
    checkpoint = torch.load(filename)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Model loaded from {filename}")
    return epoch

optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Load the trained model if it exists
if os.path.exists(MODEL_PATH):
    load_gan(generator, discriminator, optimizer_g, optimizer_d, filename=MODEL_PATH)
else:
    print("Model not found, starting with an untrained generator.")

# Load the dataset for training or augmentation
dataset = ImageFolder(root=TRAIN_PATH, transform=transform)
# Now use the trained generator to augment the dataset
augmented_dataset = AugmentedDataset(real_data=dataset, generator=generator, z_dim=100, num_fake_images=5000, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
augmented_dataloader = DataLoader(augmented_dataset, batch_size=32, shuffle=True)

def train_model(model, _dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in _dataloader:
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
        epoch_loss = running_loss / len(_dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
