import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from src.model_setup import get_configured_model
from src.fgsm_attack import execute_fgsm_attack

# Configuration
WEIGHTS_PATH = './models/resnet50_leukemia_baseline.pth'
DATA_DIR = './data/'
DEVICE = torch.device("cpu")
EPSILON = 0.05

def evaluate_full_attack():
    print("--- Initializing Dataset-Wide Red Team Attack ---")
    model = get_configured_model(weights_path=WEIGHTS_PATH, device=DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    test_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    class_names = test_dataset.classes
    
    all_adv_preds = []
    all_labels = []
    
    print(f"Deploying FGSM (Epsilon: {EPSILON}) against {len(test_dataset)} images...")
    print("Running CPU Inference (This will take a bit longer due to gradient tracking...)")
    
    # We do NOT use torch.no_grad() here because FGSM requires gradients to calculate the attack
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        # 1. Generate the adversarial batch
        adv_inputs = execute_fgsm_attack(model, inputs, labels, criterion, EPSILON, DEVICE)
        
        # 2. Get the model's predictions on the tampered batch
        # (We use no_grad here just for the final prediction to save memory)
        with torch.no_grad():
            adv_outputs = model(adv_inputs)
            _, adv_preds = torch.max(adv_outputs, 1)
            
        all_adv_preds.extend(adv_preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
            
    print("\n--- Post-Attack Clinical Evaluation Metrics ---")
    report = classification_report(all_labels, all_adv_preds, target_names=class_names, digits=4)
    print(report)

if __name__ == "__main__":
    evaluate_full_attack()