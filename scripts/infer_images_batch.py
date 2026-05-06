#!/usr/bin/env python3
# scripts/infer_images_batch.py
import argparse, os, csv
from pathlib import Path
from collections import Counter
import torch
from torchvision import transforms
from PIL import Image

def get_device(prefer=None):
    if prefer == "cpu": return torch.device("cpu")
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch, "has_mps", False): return torch.device("mps")
    return torch.device("cpu")

def build_backbone(num_classes=2, arch="resnet50"):
    from torchvision import models
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

def load_model(model_path, num_classes=2, device=None):
    device = device or get_device()
    ckpt = torch.load(model_path, map_location=device)

    # whole model
    if hasattr(ckpt, "eval") and callable(ckpt.eval):
        model = ckpt
        model.eval()
        return model.to(device)

    # dict with model_state_dict or state_dict
    if isinstance(ckpt, dict):
        state = None
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state = ckpt
        if state is None:
            raise RuntimeError("Couldn't find model state in checkpoint")

        # remove 'module.' prefix if present
        first_key = next(iter(state.keys()))
        if first_key.startswith("module."):
            new_state = {}
            for k,v in state.items():
                new_state[k.replace("module.","",1)] = v
            state = new_state

        model = build_backbone(num_classes=num_classes)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print("Warning: missing keys:", missing[:10], "...")
        if unexpected:
            print("Warning: unexpected keys:", unexpected[:10], "...")
        model.eval()
        return model.to(device)

    # unknown format
    raise RuntimeError("Unrecognized checkpoint format")

def find_images(folder):
    paths=[]
    for p in Path(folder).rglob("*"):
        if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp"):
            paths.append(str(p))
    return paths

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True, help="file or folder of images")
    p.add_argument("--out", default="results/infer_images.csv")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None, choices=[None,"cpu","mps","cuda"])
    args = p.parse_args()

    device = get_device(args.device)
    print("Device:", device)
    model = load_model(args.model, num_classes=2, device=device)
    transform = transforms.Compose([
        transforms.Resize((args.img_size,args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    if os.path.isdir(args.input):
        images = find_images(args.input)
    else:
        images = [args.input]
    print("Images to infer:", len(images))

    rows=[]
    cnt = Counter()
    batch=[]
    paths=[]
    model.eval()
    with torch.no_grad():
        for path in images:
            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                print("skip",path,e); continue
            batch.append(transform(img))
            paths.append(path)
            if len(batch) >= args.batch_size:
                x = torch.stack(batch).to(device)
                logits = model(x)
                if isinstance(logits, tuple): logits = logits[0]
                preds = torch.argmax(torch.nn.functional.softmax(logits, dim=1), dim=1).cpu().tolist()
                for pth,pr in zip(paths,preds):
                    cnt[pr]+=1
                    rows.append((pth,pr))
                batch=[]; paths=[]
        if batch:
            x=torch.stack(batch).to(device)
            logits = model(x)
            if isinstance(logits, tuple): logits = logits[0]
            preds = torch.argmax(torch.nn.functional.softmax(logits, dim=1), dim=1).cpu().tolist()
            for pth,pr in zip(paths,preds):
                cnt[pr]+=1
                rows.append((pth,pr))

    total = sum(cnt.values())
    print("Total:", total)
    for k,v in sorted(cnt.items()):
        print(f"class {k}: {v} ({100*v/total:.2f}%)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        import csv
        w=csv.writer(fh); w.writerow(["path","pred"]); w.writerows(rows)
    print("Saved", args.out)

if __name__ == "__main__":
    main()
