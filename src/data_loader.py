import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(data_dir, batch_size=32, img_size=(224, 224)):
    """
    Loads the clean leukemia dataset, applies preprocessing, 
    and returns PyTorch train/val data loaders.
    """
    
    # 1. Define Preprocessing Transformations
    preprocessing_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    
    # 2. Load the Master Dataset
    try:
        master_dataset = datasets.ImageFolder(root=data_dir, transform=preprocessing_transforms)
    except FileNotFoundError:
        raise FileNotFoundError(f"Directory not found: {data_dir}. Ensure your images are sorted here!")
        
    # 3. Perform the 80-20 Split
    train_size = int(0.8 * len(master_dataset))
    val_size = len(master_dataset) - train_size
    train_dataset, val_dataset = random_split(master_dataset, [train_size, val_size])
    
    # 4. Create the Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Loaded {len(train_dataset)} train images and {len(val_dataset)} val images.")
    print(f"Classes established: {master_dataset.classes}")
    
    return train_loader, val_loader, master_dataset.classes