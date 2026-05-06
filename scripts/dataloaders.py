# scripts/dataloaders.py
import os
import json
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
MODEL_DIR = "models"

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

def _ensure_dirs_exist(paths):
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected folder not found: {p}")

def save_class_map(class_to_idx):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "class_map.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)

def get_loaders(data_root="data/images", batch_size=BATCH_SIZE, use_testing_as_val=True):
    """
    Returns: train_loader, val_loader, test_loader
    - If data_root/val and data_root/test exist they are used.
    - Else if data_root/testing exists it will be used for both val and test.
    - Saves class->index mapping to models/class_map.json
    """
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")
    testing_dir = os.path.join(data_root, "testing")

    # ensure train exists
    _ensure_dirs_exist([train_dir])

    # decide val/test locations
    if os.path.exists(val_dir) and os.path.exists(test_dir):
        use_val = val_dir
        use_test = test_dir
    elif use_testing_as_val and os.path.exists(testing_dir):
        use_val = testing_dir
        use_test = testing_dir
    else:
        raise FileNotFoundError(f"Could not find val/test directories. Expected either '{val_dir}' & '{test_dir}', or '{testing_dir}'.")

    # Create datasets (ImageFolder expects subfolders per class)
    train_ds = ImageFolder(root=train_dir, transform=train_transform)
    val_ds   = ImageFolder(root=use_val,   transform=val_transform)
    test_ds  = ImageFolder(root=use_test,  transform=val_transform)

    # Print and save class mapping for downstream code
    class_map = train_ds.class_to_idx  # e.g. {"real":0,"synth":1} or {"fake":0,"real":1}
    print("Dataset classes (train):", train_ds.classes)
    print("Class to index mapping (train):", class_map)
    save_class_map(class_map)
    print(f"Saved class mapping to {os.path.join(MODEL_DIR, 'class_map.json')}")

    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")
    return train_loader, val_loader, test_loader

