# scripts/extract_embeddings.py
import torch, numpy as np
from torchvision import models
from scripts.dataloaders import get_loaders
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMB_DIR = "models"

def get_embedding_model():
    model = models.resnet50(pretrained=True)
    # remove final fc to get embedding vector
    model.fc = torch.nn.Identity()
    model = model.to(DEVICE)
    model.eval()
    return model

def extract_embeddings(split="train"):
    train_loader, val_loader, test_loader = get_loaders()
    loader_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = loader_map[split]
    model = get_embedding_model()
    all_embs = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            feats = model(imgs)            # (B, feat_dim)
            all_embs.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
    X = np.vstack(all_embs)
    y = np.hstack(all_labels)
    os.makedirs(EMB_DIR, exist_ok=True)
    np.save(os.path.join(EMB_DIR, f"embeddings_{split}.npy"), X)
    np.save(os.path.join(EMB_DIR, f"labels_{split}.npy"), y)
    print(f"Saved embeddings_{split}.npy  shape={X.shape}")
    return X, y

if __name__ == "__main__":
    # extract for train/val/test
    extract_embeddings("train")
    extract_embeddings("val")
    extract_embeddings("test")
