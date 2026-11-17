import torch

def apply_fgsm_noise(image, epsilon, data_grad):
    """
    The core mathematical perturbation. 
    Injects scaled gradient noise and clamps pixels to valid [0, 1] range.
    """
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def execute_fgsm_attack(model, input_tensor, true_label, criterion, epsilon, device):
    """
    Orchestrates the attack pipeline: tracks gradients, calculates the 
    classification loss, and generates the adversarial payload.
    """
    # Force PyTorch to track gradients for the input image pixels
    input_tensor.requires_grad = True
    
    # 1. Forward pass through the model
    output = model(input_tensor)
    
    # 2. Calculate the loss against the true clinical label
    loss = criterion(output, true_label)
    
    # 3. Zero out model gradients and run backward pass to get image gradients
    model.zero_grad()
    loss.backward()
    
    # 4. Extract the gradients of the input image
    data_grad = input_tensor.grad.data
    
    # 5. Generate and return the adversarial image
    perturbed_image = apply_fgsm_noise(input_tensor, epsilon, data_grad)
    
    return perturbed_image