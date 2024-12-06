import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
from pathlib import Path
import pandas as pd
import os
import shutil
from tqdm import tqdm
from architectures import FireClassifierCNN, FireClassifierViT, DeepResNet
from PIL import Image
import numpy as np

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = sorted(list(self.root_dir.glob('*.jpg')))  
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        return image, str(img_path.name)

class EnsembleModel(nn.Module):
    def __init__(self, input_size=3):
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

def load_models(device):
    cnn_model = FireClassifierCNN().to(device)
    cnn_checkpoint = torch.load('best_fire_classifier_cnn_v2_mean.pth')
    cnn_model.load_state_dict(cnn_checkpoint['model_state_dict'])
    
    vit_model = FireClassifierViT().to(device)
    vit_checkpoint = torch.load('best_fire_classifier_vit_v2_mean.pth')
    vit_model.load_state_dict(vit_checkpoint['model_state_dict'])
    
    resnet_model = DeepResNet().to(device)
    resnet_checkpoint = torch.load('best_deep_resnet_v2.pth')
    resnet_model.load_state_dict(resnet_checkpoint['model_state_dict'])
    
    ensemble = EnsembleModel(input_size=3).to(device)
    ensemble_checkpoint = torch.load('best_ensemble_model.pth')
    ensemble.load_state_dict(ensemble_checkpoint['model_state_dict'])
    
    return cnn_model, vit_model, resnet_model, ensemble

def save_uncertain_image(image_path, probability, output_dir):
    img = cv2.imread(str(image_path))
    filename = f"{probability:.3f}_{image_path.name}"
    output_path = output_dir / filename
    cv2.imwrite(str(output_path), img)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    uncertain_dir = Path('uncertain_predictions')
    uncertain_dir.mkdir(exist_ok=True)
    
    shutil.rmtree(uncertain_dir)
    uncertain_dir.mkdir()
    

    means=  [0.4487725496292114, 0.39843758940696716, 0.3449983298778534]
    stds = [0.2581987977027893, 0.2438497394323349, 0.26241636276245117]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=means, std=stds)
    ])
    
    test_dataset = TestDataset('dl2425_challenge_dataset/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    models = load_models(device)
    cnn_model, vit_model, resnet_model, ensemble = models
    
    for model in models:
        model.eval()
    
    predictions = []
    image_names = []
    
    uncertainty_threshold = 0.25  # Consider predictions with probability between 0.5 ± threshold as uncertain
    
    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc="Processing test data"):
            images = images.to(device)
            
            base_predictions = []
            for model in [cnn_model, vit_model, resnet_model]:
                outputs = model(images)
                probs = torch.sigmoid(outputs.squeeze())
                base_predictions.append(probs)
            
            stacked_preds = torch.stack(base_predictions, dim=1)
            
            ensemble_outputs = ensemble(stacked_preds)
            final_probs = torch.sigmoid(ensemble_outputs.squeeze())
            
            for prob, filename in zip(final_probs, filenames):
                prob_val = prob.item()
                predictions.append(1 if prob_val > 0.5 else 0)
                image_names.append(filename)
                
                if abs(prob_val - 0.5) < uncertainty_threshold:
                    image_path = Path('dl2425_challenge_dataset/test') / filename
                    save_uncertain_image(image_path, prob_val, uncertain_dir)
    
    submission = pd.DataFrame({
        'id': image_names,
        'class': predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    print(f"Submission file created with {len(submission)} predictions")
    print(f"Found {len(list(uncertain_dir.glob('*.jpg')))} uncertain predictions")
    print(f"Uncertain predictions saved to {uncertain_dir}")

if __name__ == '__main__':
    main()