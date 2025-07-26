import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CelebAHQDataset(Dataset):
    def __init__(self, image_size, data_dir='D:/celeba_hq_256'):
        self.data_dir = data_dir
        self.image_size = image_size
        self.image_paths = [
            os.path.join(data_dir, fname)
            for fname in os.listdir(data_dir)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {data_dir}")
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # Converts to [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizes to [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            if image.shape != torch.Size([3, self.image_size, self.image_size]):
                print(f"Warning: Image {img_path} has shape {image.shape}, expected [3, {self.image_size}, {self.image_size}]")
                return torch.zeros(3, self.image_size, self.image_size)
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros(3, self.image_size, self.image_size)

if __name__ == "__main__":
    dataset = CelebAHQDataset(image_size=64)
    sample_image = dataset[0]
    print(f"Sample image shape: {sample_image.shape}")
    print(f"Sample image min: {sample_image.min().item()}, max: {sample_image.max().item()}")