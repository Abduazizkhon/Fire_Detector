import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from pathlib import Path
import pandas as pd

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FireClassifier()

checkpoint = torch.load('best_fire_classifier.pth', map_location=device)
new_state_dict = {}
for k, v in checkpoint.items():
    name = k[7:] if k.startswith("module.") else k  # Remove 'module.' prefix if present
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model = model.to(device)
model.eval()

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(model, image_path, transform, device):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    image = transform(image).unsqueeze(0).to(device)  
    with torch.no_grad():
        output = model(image)
        prediction = 1 if output.item() > 0.5 else 0  
    return prediction

test_dir = Path('./test')  
output_file = './predictions.csv'

results = []
i = 1
for image_path in test_dir.glob("*.*"):  
    prediction = predict_image(model, image_path, test_transform, device)
    print(f"Finished: {i/1562*100}%")
    i += 1
    results.append({"id": image_path.name, "class": prediction})

df = pd.DataFrame(results)
df.to_csv(output_file, index=False)

print(f"Inference completed. Results saved to {output_file}")