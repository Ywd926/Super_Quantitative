import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import os

class SimpleCNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleCNN, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        feature_map_size = input_size // 2 // 2
        self.fc1 = nn.Linear(64 * feature_map_size * feature_map_size, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes: positive and negative

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Custom dataset class
class DropletDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB') # Ensure 3 channels
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
        except FileNotFoundError:
            print(f"Warning: File not found {self.image_paths[idx]}")
            return torch.zeros((3, 6, 6)), torch.tensor(0) # Dummy data matching expected transform
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            return torch.zeros((3, 6, 6)), torch.tensor(0)

def get_data_transforms():
    # Transformations for the training dataset
    return transforms.Compose([
        transforms.ToTensor()
    ])

def load_and_prepare_data(pos_dir='pos', neg_dir='neg', transform=None):
    if not os.path.exists(pos_dir):
        os.makedirs(pos_dir)
        print(f"Created directory: {pos_dir}")
    if not os.path.exists(neg_dir):
        os.makedirs(neg_dir)
        print(f"Created directory: {neg_dir}")
    pos_images = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir) if f.endswith('.png')]
    neg_images = [os.path.join(neg_dir, f) for f in os.listdir(neg_dir) if f.endswith('.png')]
    if not pos_images and not neg_images:
        print("Warning: No images found in 'pos' or 'neg' directories. Training might fail or be ineffective.")
    image_paths = pos_images + neg_images
    labels = [1] * len(pos_images) + [0] * len(neg_images) # 1 for positive, 0 for negative
    if not image_paths: # No images to train on
        return None, None
    dataset = DropletDataset(image_paths, labels, transform=transform if transform else get_data_transforms())
    return dataset, image_paths

if __name__ == '__main__':
    transform = get_data_transforms()
    dataset, _ = load_and_prepare_data(transform=transform)
    if dataset is None:
        print("Without an available dataset, the model cannot be trained.")
    else:
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        input_size = 30
        model = SimpleCNN(input_size=input_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 50
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
        print("Model training completed.")

