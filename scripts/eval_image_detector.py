#!/usr/bin/env python3
# scripts/eval_image_detector.py
import os, argparse, csv
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torchvision import transforms
from PIL import Image

def get_device(prefer=None):
    if prefer == "cpu": return torch.device("cpu")
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch, "has_mps", False): return torch.device("mps")
    return torch.device("cpu")

def build_backbone(num_classes=2, arch="resnet50", device=None):
    # build a standard backbone matching your training script (resnet50 used in training)
    from torchvision import models
    if arch.startswith("resnet"):
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        # fallback: small classifier
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

def load_model(model_path, num_classes=2, device=None):
    device = device or get_device()
    ckpt = torch.load(model_path, map_location=device)

    # Case 1: whole model saved
    if hasattr(ckpt, "eval") and callable(ckpt.eval):
        model = ckpt
        model.eval()
        return model.to(device)

    # Case 2: checkpoint dict with 'model_state_dict'
    if isinstance(ckpt, dict):
        # some checkpoints use 'model_state_dict', some use 'state_dict'
        state = None
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            # maybe they directly saved state_dict but inside keys like 'model'
            # try to guess
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                state = ckpt

        # classes optional
        classes = ckpt.get("classes") if isinstance(ckpt.get("classes"), (list,tuple)) else None

        if state is None:
            raise RuntimeError("Couldn't find a model state in checkpoint")

        # build backbone and load
        model = build_backbone(num_classes=num_classes)
        # If keys are prefixed (e.g. 'module.'), try to strip them
        sd_keys = list(state.keys())
        if sd_keys and sd_keys[0].startswith("module."):
            new_state = {}
            for k,v in state.items():
                new_state[k.replace("module.","",1)] = v
            state = new_state

        # Try loading; allow missing/unexpected keys but inform user
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print("Warning: missing keys in state_dict:", missing[:10], "...")
        if unexpected:
            print("Warning: unexpected keys in state_dict:", unexpected[:10], "...")
        model.eval()
        return model.to(device)

    # Case 3: pure state_dict saved (not wrapped) -> build backbone and load
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        model = build_backbone(num_classes=num_classes)
        model.load_state_dict(ckpt)
        model.eval()
        return model.to(device)

    raise RuntimeError("Unrecognized checkpoint format at: " + str(model_path))


def image_paths_from_dir(data_dir, classes=None):
    data_dir = Path(data_dir)
    items = []
    if classes:
        for c in classes:
            p = data_dir / c
            if p.exists():
                for f in p.iterdir():
                    if f.is_file() and f.suffix.lower() in (".jpg",".jpeg",".png",".bmp"):
                        items.append((str(f), c))
    else:
        for cdir in data_dir.iterdir():
            if cdir.is_dir():
                for f in cdir.iterdir():
                    if f.is_file() and f.suffix.lower() in (".jpg",".jpeg",".png",".bmp"):
                        items.append((str(f), cdir.name))
    return items

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, help="dir with class subfolders (e.g., val/real val/synth) or flat")
    p.add_argument("--classes", nargs="+", default=["real","synth"])
    p.add_argument("--out", default="results/image_eval_preds.csv")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None, choices=[None,"cpu","mps","cuda"])
    args = p.parse_args()

    device = get_device(args.device)
    print("Using device:", device)
    model = load_model(args.model, num_classes=len(args.classes), device=device)

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    items = image_paths_from_dir(args.data, classes=args.classes)
    if not items:
        # maybe directory is flat with images and labels in CSV not provided — handle gracefully
        # collect all images in directory (no labels)
        items = []
        for p in Path(args.data).rglob("*"):
            if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp"):
                items.append((str(p),"unknown"))
    print("Found", len(items), "images")

    y_true, y_pred = [], []
    rows = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(items), args.batch_size):
            batch = items[i:i+args.batch_size]
            imgs = []
            for path,cls in batch:
                try:
                    img = Image.open(path).convert("RGB")
                except Exception as e:
                    print("Skipping", path, "err:", e)
                    continue
                imgs.append(transform(img))
                # provide label index if known; otherwise -1
                y_true.append(args.classes.index(cls) if cls in args.classes else -1)
            if not imgs: continue
            x = torch.stack(imgs).to(device)
            logits = model(x)
            if isinstance(logits, tuple): logits = logits[0]
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            for (path,cls),pred in zip(batch,preds):
                rows.append((path, cls, int(pred)))

    # filter out unknown labels when computing report
    y_true_filtered = [yt for yt in y_true if yt>=0]
    if y_true_filtered:
        # map y_pred accordingly
        y_pred_for_report = [p for yt,p in zip(y_true,y_pred) if yt>=0]
        print("\nClassification report:")
        print(classification_report(y_true_filtered, y_pred_for_report, target_names=args.classes, digits=4))
        print("Confusion matrix:\n", confusion_matrix(y_true_filtered, y_pred_for_report))
    else:
        print("No ground-truth labels found; saved per-image predictions only.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path","label","pred"])
        w.writerows(rows)
    print("Saved predictions to", args.out)

if __name__ == "__main__":
    main()
