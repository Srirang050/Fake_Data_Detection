# scripts/train_resnet.py
import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from scripts.dataloaders import get_loaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 8
LR = 1e-4
MODEL_OUT = "models/image_resnet50.pt"

def train():
    train_loader, val_loader, _ = get_loaders()
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: real / synth
    model = model.to(DEVICE)

    # Freeze early layers optionally (uncomment if you want)
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    best_val = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start = time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total
        train_loss = running_loss / total

        # validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total if val_total>0 else 0.0
        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{EPOCHS}  time:{elapsed:.1f}s  train_loss:{train_loss:.4f} train_acc:{train_acc:.4f}  val_acc:{val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)
            print("Saved best model:", MODEL_OUT)

    print("Training done. Best val acc:", best_val)

if __name__ == "__main__":
    train()
