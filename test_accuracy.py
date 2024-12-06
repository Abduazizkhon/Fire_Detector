import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from architectures import *
from tqdm import tqdm

class FireDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        
        fire_path = self.root_dir / '1'
        for img_path in fire_path.glob('*.*'):
            self.image_paths.append(str(img_path))
            self.labels.append(1)
        
        no_fire_path = self.root_dir / '0'
        for img_path in no_fire_path.glob('*.*'):
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

def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model_path, val_dir, device='cuda'):
    if 'vision_transformer' in model_path:
        model = VisionTransformer(
            image_size=224,
            patch_size=16,
            in_channels=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            dropout=0.1
        ).to(device) 
    elif 'deep_resnet' in model_path:
        model = DeepResNet(
            layers=[3, 4, 23, 3],  # ResNet101 configuration
            groups=32,  
            width_per_group=8
        ).to(device)
    elif 'deep_cnn_with_attention' in model_path:
        model = DeepCNNWithAttention()
    elif 'fire_classifier_vit' in model_path:
        model = FireClassifierViT()
    elif 'fire_classifier_cnn' in model_path:
        model = FireClassifierCNN()
    elif 'resnext101' in model_path:
        model = ResNeXt101()
    elif 'resnet50' in model_path:
        model = FireClassifierResNet50()
    else:
        print("Model not found")
        return
    
    checkpoint = torch.load(model_path)
    if 'model_state_dict' in checkpoint:
        # If saved as a checkpoint dictionary
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If saved as just the state dict
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = FireDataset(val_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    all_preds = []
    all_labels = []
    
    print("Starting evaluation...")
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            outputs = model(images)
            preds = (outputs.squeeze() > 0.5).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    plot_confusion_matrix(cm)
    
    true_neg, false_pos, false_neg, true_pos = cm.ravel()
    no_fire_acc = true_neg / (true_neg + false_pos)
    fire_acc = true_pos / (true_pos + false_neg)
    
    print("\nPer-class Accuracy:")
    print(f"No Fire Detection Accuracy: {no_fire_acc:.4f}")
    print(f"Fire Detection Accuracy: {fire_acc:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'no_fire_acc': no_fire_acc,
        'fire_acc': fire_acc
    }

if __name__ == "__main__":
    # model_path = 'best_fire_classifier.pth'

    choices = ['vision_transformer', # index 0
                'deep_resnet', # index 1
                'deep_cnn_with_attention',  # index 2
                'fire_classifier_vit',  # index 3
                'fire_classifier_cnn',  # index 4
                'resnext101', # index 5
                'resnet50', # index 6
                ]
    current_model = choices[4]

    if current_model == 'vision_transformer':
        model_name = 'vision_transformer'

    elif current_model == 'deep_resnet':
        model_name = 'deep_resnet'

    elif current_model == 'deep_cnn_with_attention':
        model_name = 'deep_cnn_with_attention'

    elif current_model == 'fire_classifier_vit':
        model_name = 'fire_classifier_vit'

    elif current_model == 'fire_classifier_cnn':
        model_name = 'fire_classifier_cnn'
    
    elif current_model == 'resnext101':
        model_name = 'resnext101'
    
    elif current_model == 'resnet50':
        model_name = 'resnet50'

    # model_path=f'best_{model_name}_v2.pth' 

    # model_path = f"best_{model_name}_v2.pth"
    model_path = f"best_{model_name}_v2_mean.pth"
    print(f"Model path: {model_path}")
    val_dir = 'dl2425_challenge_dataset/val'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    metrics = evaluate_model(model_path, val_dir, device)

    print("\nMetrics:")
    print(metrics)