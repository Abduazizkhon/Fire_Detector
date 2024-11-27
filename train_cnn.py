import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

## this is just our first version using normal CNN, the purpose was just to setup the project
## our optimizations will be on this archicture (and maybe also the transformations of preprocessing)
class FireClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class FireDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        # self.root_dir = root_dir
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        
        # Load fire images (label 1)
        fire_path = self.root_dir / '1'
        # fire_path = f"{self.root_dir}/1"
        for img_path in fire_path.glob('*.*'):
        # for img_path in os.listdir(fire_path):
            self.image_paths.append(str(img_path))
            self.labels.append(1)
        
        # Load non-fire images (label 0)
        no_fire_path = self.root_dir / '0'
        # no_fire_path = f"{self.root_dir}/0"
        for img_path in no_fire_path.glob('*.*'):
        # for img_path in os.listdir(no_fire_path):
            self.image_paths.append(str(img_path))
            self.labels.append(0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((254, 254)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((254, 254)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            total_loss += loss.item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)
    return accuracy, avg_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_val_acc = 0.0
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_loss += loss.item()
        
        train_accuracy = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)

        val_accuracy, val_loss = evaluate_model(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_fire_classifier.pth')
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        print('-' * 60)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # train_dir = 'dl2425_challenge_dataset/train'
    # val_dir = 'dl2425_challenge_dataset/val'

    base_dir = Path('dl2425_challenge_dataset')
    train_dir = base_dir / 'train'
    val_dir = base_dir / 'val'
    
    print(f"Training directory: {train_dir}")
    print(f"Validation directory: {val_dir}")
    
    try:
        train_dataset = FireDataset(train_dir, transform=train_transform)
        print(f"Successfully loaded {len(train_dataset)} training images")
        
        val_dataset = FireDataset(val_dir, transform=val_transform)
        print(f"Successfully loaded {len(val_dataset)} validation images")
        
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)  # Set num_workers to 0 for debugging
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=1)
 
    
    # # Create datasets
    # train_dataset = FireDataset(train_dir, transform=train_transform)
    # val_dataset = FireDataset(val_dir, transform=val_transform)
      
    model = FireClassifier().to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=50,
        device=device
    )

if __name__ == '__main__':
    main()