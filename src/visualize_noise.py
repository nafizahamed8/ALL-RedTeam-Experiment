import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from src.model_setup import get_configured_model
from src.fgsm_attack import execute_fgsm_attack

# Configuration
WEIGHTS_PATH = './models/resnet50_leukemia_baseline.pth'
DATA_DIR = './data/'
DEVICE = torch.device("cpu")
EPSILON = 0.05

def visualize_attack_footprint():
    print("Loading model and grabbing a sample image...")
    model = get_configured_model(weights_path=WEIGHTS_PATH, device=DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    # Setup data loading for a single image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    test_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    # Batch size of 1 so we just grab the very first image
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True) 
    class_names = test_dataset.classes
    
    # Grab one image and its true label
    image_tensor, true_label = next(iter(test_loader))
    image_tensor = image_tensor.to(DEVICE)
    true_label = true_label.to(DEVICE)
    
    print("Executing FGSM attack...")
    # Generate the adversarial image
    adv_image = execute_fgsm_attack(model, image_tensor, true_label, criterion, EPSILON, DEVICE)
    
    print("Extracting mathematical noise...")
    # Calculate the raw forensic footprint
    raw_noise = adv_image - image_tensor
    
    # --- Visualization Formatting ---
    # Convert tensors to numpy format for Matplotlib (Height, Width, Channels)
    orig_display = image_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
    adv_display = adv_image[0].detach().cpu().permute(1, 2, 0).numpy()
    
    # AMPLIFY the noise so human eyes can see it (Normalize it to a 0-1 scale)
    noise_display = raw_noise[0].detach().cpu().permute(1, 2, 0).numpy()
    noise_display = (noise_display / EPSILON) * 0.5 + 0.5 
    
    # Plotting the 1x3 Grid
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(orig_display)
    axes[0].set_title(f"Original Image\nTrue Class: {class_names[true_label.item()]}")
    axes[0].axis('off')
    
    axes[1].imshow(noise_display)
    axes[1].set_title(f"FGSM Noise Footprint\n(Amplified for Visibility)")
    axes[1].axis('off')
    
    # Get the model's new prediction to prove the attack worked
    with torch.no_grad():
        pred_logits = model(adv_image)
        pred_class = torch.argmax(pred_logits, dim=1).item()
        
    axes[2].imshow(adv_display)
    axes[2].set_title(f"Adversarial Image ($\epsilon$={EPSILON})\nPredicted Class: {class_names[pred_class]}")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save to your paper directory
    save_path = './paper/noise_footprint.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    visualize_attack_footprint()