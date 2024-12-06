import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from train import FireDataset
from architectures import FireClassifierCNN, FireClassifierViT, DeepResNet
from ensemble_train import EnsembleModel, load_base_models
from tqdm import tqdm
import numpy as np

def calculate_metrics(predictions, labels):
    total = len(labels)
    correct = (predictions == labels).sum()
    accuracy = 100 * correct / total
    
    true_pos = ((predictions == 1) & (labels == 1)).sum()
    false_pos = ((predictions == 1) & (labels == 0)).sum()
    false_neg = ((predictions == 0) & (labels == 1)).sum()
    
    precision = true_pos / (true_pos + false_pos + 1e-8)
    recall = true_pos / (true_pos + false_neg + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return accuracy, precision * 100, recall * 100, f1 * 100

def evaluate_ensemble(ensemble_model, base_models, val_loader, device):
    cnn_model, vit_model, resnet_model = base_models
    ensemble_model.eval()
    
    all_predictions = []
    all_labels = []
    individual_predictions = {'CNN': [], 'ViT': [], 'ResNet': []}
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            batch_base_preds = []
            
            for i, model in enumerate([cnn_model, vit_model, resnet_model]):
                outputs = model(images)
                probs = torch.sigmoid(outputs.squeeze())
                batch_base_preds.append(probs)
                
                model_preds = (probs > 0.5).float().cpu().numpy()
                individual_predictions[list(individual_predictions.keys())[i]].extend(model_preds)
            
            ensemble_input = torch.stack(batch_base_preds, dim=1)
            ensemble_output = ensemble_model(ensemble_input)
            ensemble_preds = (torch.sigmoid(ensemble_output.squeeze()) > 0.5).float()
            
            all_predictions.extend(ensemble_preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    models_metrics = {}
    models_metrics['Ensemble'] = calculate_metrics(np.array(all_predictions), np.array(all_labels))
    
    for model_name, preds in individual_predictions.items():
        models_metrics[model_name] = calculate_metrics(np.array(preds), np.array(all_labels))
    
    print("\nModel Performance Metrics:")
    print(f"{'Model':<10}{'Accuracy':>10}{'Precision':>12}{'Recall':>10}{'F1':>10}")
    print("-" * 52)
    
    for model_name, metrics in models_metrics.items():
        acc, prec, rec, f1 = metrics
        print(f"{model_name:<10}{acc:>10.2f}{prec:>12.2f}{rec:>10.2f}{f1:>10.2f}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    means =  [0.4487725496292114, 0.39843758940696716, 0.3449983298778534]
    stds = [0.2581987977027893, 0.2438497394323349, 0.26241636276245117]
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.Normalize(mean=means, std=stds)
    ])
    
    val_dataset = FireDataset('dl2425_challenge_dataset/val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    base_models = load_base_models(device)
    ensemble = EnsembleModel(input_size=3).to(device)
    checkpoint = torch.load('best_ensemble_model.pth')
    ensemble.load_state_dict(checkpoint['model_state_dict'])
    
    evaluate_ensemble(ensemble, base_models, val_loader, device)

if __name__ == '__main__':
    main()