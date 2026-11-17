import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from src.model_setup import get_configured_model

WEIGHTS_PATH = './models/resnet50_leukemia_baseline.pth'
DATA_DIR = './data/'  # Make sure 'all' and 'hem' folders are directly inside here!
DEVICE = torch.device("cpu") # CPU inference is perfect here

def evaluate_test_set():
    print("--- Initializing Final Clinical Evaluation ---")
    
    # 1. Load the fine-tuned model
    model = get_configured_model(weights_path=WEIGHTS_PATH, device=DEVICE)
    
    # 2. Setup the exact same preprocessing used in Colab
    preprocessing_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # 3. Load the pure test dataset
    test_dataset = datasets.ImageFolder(root=DATA_DIR, transform=preprocessing_transforms)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    class_names = test_dataset.classes
    
    print(f"Loaded {len(test_dataset)} pure test images.")
    print("Running CPU Inference (Grab a coffee, this may take 5-10 minutes on a laptop...)")
    
    all_preds = []
    all_labels = []
    
    # 4. Run the evaluation
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    print("\n--- Final Clinical Evaluation Metrics ---")
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

if __name__ == "__main__":
    evaluate_test_set()