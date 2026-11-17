import torch
import torch.nn as nn
from torchvision import models

def get_configured_model(weights_path=None, device='cpu'):
    """
    Loads the ResNet-50 architecture, configures the classification head 
    for our 2 specific classes, and injects our fine-tuned weights.
    """
    print("Initializing ResNet-50 architecture...")
    
    # Load the base model skeleton 
    # (We use weights=None here because we are about to load our own fine-tuned weights!)
    model = models.resnet50(weights=None)
    
    # Modify the Classification Head for 2 classes (all vs. hem)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    # Load custom weights if a path is provided
    if weights_path:
        try:
            print(f"Loading fine-tuned weights from: {weights_path}")
            # map_location ensures it loads safely whether your local machine has a GPU or not
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print("Weights successfully injected!")
        except FileNotFoundError:
            print(f"WARNING: Weights file not found at {weights_path}. Model will be untrained.")
    
    # Push model to the correct hardware and set to Evaluation mode
    model = model.to(device)
    model.eval() 
    
    return model