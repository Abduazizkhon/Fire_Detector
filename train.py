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
import random
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from architectures import *
from torch.multiprocessing import freeze_support
from torch.utils.tensorboard import SummaryWriter

class FireDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_training=True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_training = is_training
        
        self.image_paths = []
        self.labels = []
        
        fire_path = self.root_dir / '1'
        no_fire_path = self.root_dir / '0'
        
        fire_images = list(fire_path.glob('*.*'))
        no_fire_images = list(no_fire_path.glob('*.*'))
        
        self.image_paths.extend([str(path) for path in fire_images])
        self.labels.extend([1] * len(fire_images))
        
        self.image_paths.extend([str(path) for path in no_fire_images])
        self.labels.extend([0] * len(no_fire_images))
        
        self.pos_weight = len(no_fire_images) / len(fire_images)

        if self.is_training: ### shuffle the dataset
            combined = list(zip(self.image_paths, self.labels))
            random.shuffle(combined)
            self.image_paths[:], self.labels[:] = zip(*combined)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

class RandAugment:
    def __init__(self, n, m):
        self.n = n  # number of augmentation transformations to apply
        self.m = m  # magnitude for all the transformations
        self.augment_list = [
            transforms.RandomRotation(degrees=30),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        ]

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op in ops:
            img = op(img)
        return img
    



def calculate_mean_std(data_dir):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = FireDataset(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, num_workers=0)
    
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    for images, _ in loader:
        channels_sum += torch.mean(images, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(images ** 2, dim=[0, 2, 3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    
    return mean.tolist(), std.tolist()

# train_dir = 'dl2425_challenge_dataset/train'
# means, stds = calculate_mean_std(train_dir)
# # print(f"Means: {means}")
# # print(f"Stds: {stds}")


# train_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),  # Larger initial size for better cropping
#     RandAugment(n=2, m=9),  # Apply 2 random augmentations with magnitude 9
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     transforms.Normalize(mean=means, std=stds)
# ])

# # Validation transforms with center crop
# val_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     # transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     transforms.Normalize(mean=means, std=stds) ## our dataset mean and std
# ])


# # Modified transforms for ViT
# train_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),  # ViT standard input size
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     transforms.Normalize(mean=means, std=stds)  # Normalize with dataset mean and std
# ])

# val_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     transforms.Normalize(mean=means, std=stds)
# ])


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_path='best_fire_classifier.pth'):
    writer = SummaryWriter(save_path.replace('.pth', '_logs'))
    scaler = GradScaler()
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_true_pos = 0
        train_false_pos = 0
        train_false_neg = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if scheduler:
                scheduler.step()


            threshold = 0.5
            probabilities = torch.sigmoid(outputs.squeeze())
            predicted = (probabilities > threshold).float()

            # predicted = (outputs.squeeze() > 0.0).float()


            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_true_pos += ((predicted == 1) & (labels == 1)).sum().item()
            train_false_pos += ((predicted == 1) & (labels == 0)).sum().item()
            train_false_neg += ((predicted == 0) & (labels == 1)).sum().item()
            
            train_loss += loss.item()
            
            if i % 10 == 0:  # Update every 10 batches
                writer.add_scalar('Training/BatchLoss', loss.item(), epoch * len(train_loader) + i)
        
        train_accuracy = 100 * train_correct / train_total
        train_precision = train_true_pos / (train_true_pos + train_false_pos + 1e-8)
        train_recall = train_true_pos / (train_true_pos + train_false_neg + 1e-8)
        train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall + 1e-8)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_true_pos = 0
        val_false_pos = 0
        val_false_neg = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                
                probabilities = torch.sigmoid(outputs.squeeze())
                predicted = (probabilities > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_true_pos += ((predicted == 1) & (labels == 1)).sum().item()
                val_false_pos += ((predicted == 1) & (labels == 0)).sum().item()
                val_false_neg += ((predicted == 0) & (labels == 1)).sum().item()
                
                val_loss += loss.item()
        
        val_accuracy = 100 * val_correct / val_total
        val_precision = val_true_pos / (val_true_pos + val_false_pos + 1e-8)
        val_recall = val_true_pos / (val_true_pos + val_false_neg + 1e-8)
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-8)
        
        writer.add_scalars('Metrics/Loss', {
            'train': train_loss/len(train_loader),
            'val': val_loss/len(val_loader)
        }, epoch)
        
        writer.add_scalars('Metrics/Accuracy', {
            'train': train_accuracy,
            'val': val_accuracy
        }, epoch)
        
        writer.add_scalars('Metrics/F1', {
            'train': train_f1,
            'val': val_f1
        }, epoch)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train - Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%, F1: {train_f1:.4f}')
        print(f'Val - Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%, F1: {val_f1:.4f}')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_accuracy
            }, save_path)
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    writer.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')




    # Calculate mean and std on training data
    train_dir = 'dl2425_challenge_dataset/train'
    means, stds = calculate_mean_std(train_dir)
    print(f"Means: {means}")
    print(f"Stds: {stds}")

    # # with RandAugment (strong augmentations, but seems not have a big impact on the model)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Larger initial size for better cropping
        RandAugment(n=2, m=9),  # Apply 2 random augmentations with magnitude 9
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=means, std=stds)
    ])

    # Validation transforms with center crop
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=means, std=stds) ## our dataset mean and std
    ])



    # # Modified transforms for ViT
    # train_transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((224, 224)),  # ViT standard input size
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(15),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    #     transforms.ToTensor(),
    #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     transforms.Normalize(mean=means, std=stds)  # Normalize with dataset mean and std
    # ])

    # val_transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     transforms.Normalize(mean=means, std=stds)
    # ])
    
    base_dir = Path('dl2425_challenge_dataset')
    train_dir = base_dir / 'train'
    val_dir = base_dir / 'val'
    
    train_dataset = FireDataset(train_dir, transform=train_transform, is_training=True)
    val_dataset = FireDataset(val_dir, transform=val_transform, is_training=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )


    # model = FireClassifierCNN().to(device) ## 86%, 88%, 91% in epoch 1,2,3, 97% at 33 epochs, very fast to train; best: 98.05%

    ########### Uncomment For using our Vision Transformer Model, not perfoming very good tough (and also heavy to train) | 76% accuracy at 8 epochs
    # model = VisionTransformer(
    #     image_size=224,
    #     patch_size=16,
    #     in_channels=3,
    #     embed_dim=768,
    #     depth=12,
    #     num_heads=12,
    #     mlp_ratio=4,
    #     dropout=0.1
    # ).to(device)

    ###### Pretrained vision transformer
    # model = FireClassifierViT().to(device) ### 330MB pretrained model, 98.24% accuracy after epoch1, 98.88% after 5 epochs; best:98.94%
    #### with new mean and std (of dataset) and IMagenetV1 weights. 98.53 98.43 98.56% in 3 epochs
    ##### with newest weights weights=ViT_B_16_Weights.DEFAULT   98.56 98.3  98.91 98.21 in 4epochs

    ########### OR for Deep ResNet (ResNet101-like with improvements), very slow to train, 75% 84% 87% in epoch 1,2,3, 97% at 24 epochs
    ########   83, 85, 92% in epoch 1,2,3 with this new training code 
    # model = DeepResNet(
    #     layers=[3, 4, 23, 3],  # ResNet101 configuration
    #     groups=32,  # ResNeXt-like grouped convolutions
    #     width_per_group=8
    # ).to(device)

    # model = FireClassifierResNet152().to(device)  ### Resnet 152 ### 90.84% after 10 epochs, very slow 

    ######### CNN with attention, faster to train than visionTransformer, 95-97 accuracy after 26 epochs ### 
    # model = DeepCNNWithAttention().to(device)  ## 89% in first 3 epochs

    model = FireClassifierResNet50().to(device)  ## 

    # model = ResNeXt101(pretrained=True, unfreeze_last_n_layers=2).to(device)  ## 84% 77% 86% 92%  :final: 96.76% 16 epochs
    ### with pretrained weight and no freeze: 96.28%  epoch1 
    ### freezing all cnn layers:   97.5% early stopping at 18 epochs
    
    # Use weighted BCE loss
    #criterion = nn.BCELoss(pos_weight=torch.tensor([train_dataset.pos_weight]).to(device))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([train_dataset.pos_weight]).to(device))
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.01)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-2,  #### initially 1e-3
        epochs=100,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )


    if isinstance(model, VisionTransformer):
        model_name = 'vision_transformer'
    elif isinstance(model, DeepResNet):
        model_name = 'deep_resnet'
    elif isinstance(model, DeepCNNWithAttention):
        model_name = 'deep_cnn_with_attention'
    elif isinstance(model, FireClassifierViT):
        model_name = 'fire_classifier_vit'
    elif isinstance(model, ResNeXt101):
        model_name = 'resnext101'
    elif isinstance(model, FireClassifierResNet152):
        model_name = 'resnet152'
    elif isinstance(model, FireClassifierCNN):
        model_name = 'fire_classifier_cnn'
    elif isinstance(model, FireClassifierResNet50):
        model_name = 'resnet50'
    else:
        print("Model not found")
        return
    
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=200,
        device=device,
        save_path=f'best_{model_name}_v2_mean.pth'
    )

if __name__ == '__main__':
    main()