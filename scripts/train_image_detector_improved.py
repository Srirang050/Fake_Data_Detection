#!/usr/bin/env python3
"""
train_image_detector_improved.py

Usage example:
python scripts/train_image_detector_improved.py \
  --data data/images \
  --out models/image_fake_detector \
  --epochs 6 \
  --batch-size 32 \
  --img-size 224 \
  --num-workers 0 \
  --freeze-backbone

Assumes data directory layout:
data/images/train/real/*.jpg
data/images/train/synth/*.jpg
data/images/val/real/*.jpg
data/images/val/synth/*.jpg

If you only have data/images/real and data/images/fake, the script expects you to
split into train/val beforehand (see your earlier commands).
"""
import argparse, os, json, time
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # allow truncated images to load

# ---------- Safe ImageFolder (prevents a single bad file from crashing training) ----------
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return sample, target
        except Exception as e:
            # Print warning and return a dummy tensor to keep batch shape consistent
            print(f"WARNING: unable to load image {path!s}: {e}")
            # create a dummy image (3 x H x W) zeros - pick same size as transform target later
            # We'll create a CPU tensor placeholder; downstream code expects Tensor
            H = W = args.img_size
            return torch.zeros(3, H, W), target

# ---------- Helpers ----------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS support (Mac Apple Silicon)
    if getattr(torch, "has_mps", False) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def build_model(num_classes=2, freeze_backbone=False):
    model = models.resnet50(pretrained=True)
    # replace fc
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes)
    if freeze_backbone:
        # freeze all except fc
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
    return model

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += imgs.size(0)
        pbar.set_postfix(loss=running_loss/total, acc=running_correct/total)
    return running_loss / total, running_correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="eval", leave=False)
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total += imgs.size(0)
            pbar.set_postfix(loss=running_loss/total, acc=running_correct/total)
    return running_loss / total, running_correct / total

# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to images root (train/val subfolders expected)")
    parser.add_argument("--out", required=True, help="Output dir to save model")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0, help="Set 0 on mac to avoid multiprocessing issues")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone and train head only")
    parser.add_argument("--device", default=None, help="override device (cpu|cuda|mps)")
    args = parser.parse_args()

    # allow transforms to reference img size
    img_size = args.img_size

    # transforms: remove hue as discussed; keep brightness/contrast/saturation
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10),  # no hue
        transforms.RandomRotation(6),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    data_root = Path(args.data)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit(f"Expected {train_dir} and {val_dir} to exist. Create train/val splits first.")

    # Use SafeImageFolder
    train_ds = SafeImageFolder(str(train_dir), transform=train_tf)
    val_ds   = SafeImageFolder(str(val_dir), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=False)

    device = torch.device(args.device) if args.device else get_device()
    print("Using device:", device)

    # build model
    num_classes = len(train_ds.classes)
    model = build_model(num_classes=num_classes, freeze_backbone=args.freeze_backbone)
    model = model.to(device)

    # optimizer -> only parameters that require grad
    opt_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.out, exist_ok=True)
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = Path(args.out) / "best_model.pth"
            torch.save({"model_state_dict": model.state_dict(),
                        "classes": train_ds.classes}, save_path)
            print("Saved best model to", save_path)

    total_time = time.time() - start_time
    # final save
    final_path = Path(args.out) / "final_model.pth"
    torch.save({"model_state_dict": model.state_dict(), "classes": train_ds.classes}, final_path)
    with open(Path(args.out) / "train_history.json", "w") as fh:
        json.dump({"history": history, "best_val_acc": best_val_acc, "total_time_sec": total_time}, fh, indent=2)
    print("Training complete. Best val acc:", best_val_acc)
    print("Saved final model to", final_path)
