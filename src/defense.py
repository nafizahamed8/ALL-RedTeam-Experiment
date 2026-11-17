import torch
import torch.optim as optim
import torch.nn.functional as F
from evaluate_attack import fgsm_attack

def train_adversarial(model, train_loader, device, epochs, epsilon, lr):
    print("\n--- Starting Adversarial Defense Training ---")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            images.requires_grad = True
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            model.zero_grad()
            loss.backward()
            
            adv_images = fgsm_attack(images, epsilon, images.grad.data).detach()
            combined_images = torch.cat((images.detach(), adv_images), dim=0)
            combined_labels = torch.cat((labels, labels), dim=0)
            
            optimizer.zero_grad()
            combined_outputs = model(combined_images)
            train_loss = F.cross_entropy(combined_outputs, combined_labels)
            train_loss.backward()
            optimizer.step()
            
            running_loss += train_loss.item()
            _, predicted = torch.max(combined_outputs.data, 1)
            total += combined_labels.size(0)
            correct += (predicted == combined_labels).sum().item()
            
        acc = 100.0 * correct / total
        print(f"Defense Epoch [{epoch+1}/{epochs}] | Robust Train Acc: {acc:.2f}%")
        
    return model