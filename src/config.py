import torch

# Random Seed for Reproducibility
SEED = 42

# Hyperparameters
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPSILONS = [0.01, 0.03, 0.10]
DEFENSE_EPOCHS = 5
DEFENSE_LR = 1e-4
DEFENSE_EPSILON = 0.05

# --- UPDATED LOCAL PATHS ---
CLEAN_DATA_DIR = './data' 
MODEL_WEIGHTS_PATH = './models/resnet50_leukemia_baseline.pth'

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")