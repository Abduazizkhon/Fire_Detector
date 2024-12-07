import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from architectures import FireClassifierCNN, FireClassifierViT, DeepResNet
from train import FireDataset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from train_2 import FireClassifier


class EnsembleModel(nn.Module):
    def __init__(self, input_size=3):  # input_size is number of models
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

def load_base_models(device):
    # Load Our base CNN model (best accuracy 98.05%)
    cnn_model = FireClassifierCNN().to(device)
    cnn_checkpoint = torch.load('best_fire_classifier_cnn_v2_means.pth')
    cnn_model.load_state_dict(cnn_checkpoint['model_state_dict'])


    # ### a new resnext model with 98.78% accuracy
    # cnn_model = FireClassifier().to(device)
    # cnn_model_path = "best_model.pth"
    # cnn_checkpoint = torch.load(cnn_model_path) 
    # cnn_model.load_state_dict(cnn_checkpoint['model_state_dict'])
    
    # Load ViT transfer learning model (best accuracy 98.7%)
    vit_model = FireClassifierViT().to(device)
    vit_checkpoint = torch.load('best_fire_classifier_vit_v2_mean.pth')
    vit_model.load_state_dict(vit_checkpoint['model_state_dict'])
    
    resnet_model = DeepResNet().to(device)
    resnet_checkpoint = torch.load('best_deep_resnet.pth')
    # resnet_model.load_state_dict(resnet_checkpoint['model_state_dict'])
    resnet_model.load_state_dict(resnet_checkpoint)
    
    return cnn_model, vit_model, resnet_model

def get_model_predictions(models, dataloader, device):
    cnn_model, vit_model, resnet_model = models
    all_predictions = []
    all_labels = []
    
    for model in [cnn_model, vit_model, resnet_model]:
        model.eval()
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Getting base model predictions"):
            images = images.to(device)
            batch_predictions = []
            
            for model in [cnn_model, vit_model, resnet_model]:
                outputs = model(images)
                probs = torch.sigmoid(outputs.squeeze())
                batch_predictions.append(probs.cpu())
            
            stacked_preds = torch.stack(batch_predictions, dim=1)
            all_predictions.append(stacked_preds)
            all_labels.append(labels)
    
    return torch.cat(all_predictions), torch.cat(all_labels)

def train_ensemble(base_models, train_loader, val_loader, device, num_epochs=50):
    writer = SummaryWriter('runs/ensemble')
    
    ensemble = EnsembleModel(input_size=len(base_models)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(ensemble.parameters(), lr=0.001)
    
    print("Getting training predictions...")
    train_preds, train_labels = get_model_predictions(base_models, train_loader, device)
    print("Getting validation predictions...")
    val_preds, val_labels = get_model_predictions(base_models, val_loader, device)
    
    train_dataset = torch.utils.data.TensorDataset(train_preds, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_preds, val_labels)
    
    train_ensemble_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_ensemble_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    best_val_f1 = 0
    patience = 18
    patience_counter = 0
    
    for epoch in range(num_epochs):
        ensemble.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_ensemble_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = ensemble(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item()
        
        train_acc = 100 * train_correct / train_total
        
        ensemble.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_true_pos = 0
        val_false_pos = 0
        val_false_neg = 0
        
        with torch.no_grad():
            for inputs, labels in val_ensemble_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = ensemble(inputs)
                loss = criterion(outputs.squeeze(), labels)
                
                predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_true_pos += ((predicted == 1) & (labels == 1)).sum().item()
                val_false_pos += ((predicted == 1) & (labels == 0)).sum().item()
                val_false_neg += ((predicted == 0) & (labels == 1)).sum().item()
                
                val_loss += loss.item()
        
        val_acc = 100 * val_correct / val_total
        val_precision = val_true_pos / (val_true_pos + val_false_pos + 1e-8)
        val_recall = val_true_pos / (val_true_pos + val_false_neg + 1e-8)
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-8)
        
        print(f'Epoch {epoch+1}:')
        print(f'Train - Loss: {train_loss/len(train_ensemble_loader):.4f}, Acc: {train_acc:.2f}%')
        print(f'Val - Loss: {val_loss/len(val_ensemble_loader):.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}')
        
        writer.add_scalars('Loss', {
            'train': train_loss/len(train_ensemble_loader),
            'val': val_loss/len(val_ensemble_loader)
        }, epoch)
        writer.add_scalars('Accuracy', {
            'train': train_acc,
            'val': val_acc
        }, epoch)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': ensemble.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
            }, 'best_ensemble_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    writer.close()
    return ensemble

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = FireDataset('dl2425_challenge_dataset/train', transform=transform)
    val_dataset = FireDataset('dl2425_challenge_dataset/val', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    base_models = load_base_models(device)
    
    ensemble = train_ensemble(base_models, train_loader, val_loader, device)

if __name__ == '__main__':
    main()