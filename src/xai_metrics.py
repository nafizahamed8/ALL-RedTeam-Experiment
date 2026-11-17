import numpy as np
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from evaluate_attack import fgsm_attack

def calculate_iou(mask1, mask2, threshold=0.5):
    bin_mask1 = (mask1 > threshold).astype(np.float32)
    bin_mask2 = (mask2 > threshold).astype(np.float32)
    intersection = np.logical_and(bin_mask1, bin_mask2).sum()
    union = np.logical_or(bin_mask1, bin_mask2).sum()
    if union == 0: return 0.0
    return intersection / union

def evaluate_gradcam_iou(model, val_loader, device, epsilon):
    print("\n--- Calculating XAI Metric (IoU) ---")
    model.eval()
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    iou_scores = []
    
    for i, (images, labels) in enumerate(val_loader):
        if i > 20: break # Subset limit to save time
        
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True
        
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        targets = [ClassifierOutputTarget(label.item()) for label in labels]
        clean_grayscale_cam = cam(input_tensor=images, targets=targets)
        
        model.zero_grad()
        loss.backward()
        data_grad = images.grad.data
        adv_images = fgsm_attack(images, epsilon, data_grad)
        
        adv_grayscale_cam = cam(input_tensor=adv_images, targets=targets)
        
        for j in range(images.size(0)):
            iou = calculate_iou(clean_grayscale_cam[j], adv_grayscale_cam[j])
            iou_scores.append(iou)
            
    mean_iou = np.mean(iou_scores)
    print(f"Mean IoU between Clean and Adversarial Grad-CAM (eps={epsilon}): {mean_iou:.4f}")
    return mean_iou