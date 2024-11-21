import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

class AugmentedDataset(Dataset):
    def __init__(self, real_data, generator, z_dim=100, num_fake_images=5000, transform=None):
        self.real_data = real_data
        self.generator = generator
        self.z_dim = z_dim
        self.num_fake_images = num_fake_images
        self.transform = transform
        
        # Apply transformation to real images to convert them to tensors if necessary
        real_images = []
        for img, _ in real_data:  # real_data is an ImageFolder object
            if isinstance(img, torch.Tensor):  # Skip transformation if it's already a tensor
                real_images.append(img)
            elif self.transform:
                img = self.transform(img)
                real_images.append(img)
            else:
                real_images.append(transforms.ToTensor()(img))  # Apply ToTensor if no custom transform
        self.real_data = torch.stack(real_images)  # Convert list of images to a tensor

        # Generate fake images
        self.fake_images = []
        for _ in range(num_fake_images):
            z = torch.randn(1, self.z_dim).to('cpu')  # Random noise for Generator
            fake_image = generator(z).cpu().detach()  # Generate fake image
            self.fake_images.append(fake_image)
        self.fake_images = torch.cat(self.fake_images, dim=0)
        
        # Combine real and fake images
        self.all_images = torch.cat([self.real_data, self.fake_images], dim=0)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        image = self.all_images[idx]
        return image, 0  # Returning 0 as label for both real and fake images
