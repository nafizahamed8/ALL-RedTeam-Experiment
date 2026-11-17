import torch
import torch.nn as nn
from src.model_setup import get_configured_model
from src.fgsm_attack import execute_fgsm_attack

# Configuration
WEIGHTS_PATH = './models/resnet50_leukemia_baseline.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON = 0.05

def main():
    print("--- Initializing Local Security Pipeline ---")
    
    # 1. Load the fine-tuned model and inject the cloud weights
    model = get_configured_model(weights_path=WEIGHTS_PATH, device=DEVICE)
    
    # 2. Create a dummy tensor (simulating a normalized 224x224 cell image)
    print("\nGenerating dummy medical tensor for pipeline validation...")
    dummy_image = torch.rand(1, 3, 224, 224).to(DEVICE)
    dummy_label = torch.tensor([0]).to(DEVICE) # Simulating an 'all' (Leukemia) label
    criterion = nn.CrossEntropyLoss()
    
    # 3. Execute the isolated attack module
    print(f"Executing FGSM Attack (Epsilon: {EPSILON})...")
    adv_image = execute_fgsm_attack(model, dummy_image, dummy_label, criterion, EPSILON, DEVICE)
    
    print(f"\n[+] Integration Test Successful!")
    print(f"    Adversarial tensor generated with shape: {adv_image.shape}")
    print("--- MLOps Pipeline Ready ---")

if __name__ == "__main__":
    main()