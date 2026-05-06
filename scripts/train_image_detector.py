#!/usr/bin/env python3
"""
train_image_detector.py
Fine-tune a pretrained CNN (EfficientNet-B0) for fake-vs-real image detection.
"""
import os, time, argparse
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/images", help="Dataset root (expects subfolders real/ fake/)")
    p.add_argument("--out", default="models/image_fake_detector", help="Where to save model")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    tf_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    tf_val = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    data_train = datasets.ImageFolder(os.path.join(args.data, "train"), transform=tf_train)
    data_val   = datasets.ImageFolder(os.path.join(args.data, "val"), transform=tf_val)
    train_loader = DataLoader(data_train, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader   = DataLoader(data_val, batch_size=args.batch, shuffle=False, num_workers=2)

    # Model
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)
    model.to(device)

    # Loss / Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0
    os.makedirs(args.out, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss, total_correct = 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            total_correct += (out.argmax(1) == labels).sum().item()
        train_acc = total_correct / len(train_loader.dataset)

        # Validation
        model.eval()
        val_correct, val_loss = 0, 0
        preds, gts = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                val_loss += loss.item() * imgs.size(0)
                p = out.argmax(1)
                preds.extend(p.cpu().numpy())
                gts.extend(labels.cpu().numpy())
                val_correct += (p == labels).sum().item()
        val_acc = val_correct / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{args.epochs}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
            print("✅ Saved new best model")

    # final report
    print("\nValidation classification report:")
    print(classification_report(gts, preds, target_names=data_train.classes))
    print("Confusion matrix:\n", confusion_matrix(gts, preds))

if __name__ == "__main__":
    main()
