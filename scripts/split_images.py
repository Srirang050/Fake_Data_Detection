# scripts/split_images.py
import os, shutil, random
from pathlib import Path

random.seed(42)

# EDIT THESE if your unzipped dataset uses different folder names
SRC_REAL  = "data/images/real"        # path where real images are located after unzip
SRC_SYNTH = "data/images    /synthetic"  # path where synthetic images are located after unzip

DST_ROOT  = "data/images"             # destination root for train/val/test splits
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}

def make_dest():
    for split in SPLITS:
        for cls in ["real","synth"]:
            Path(f"{DST_ROOT}/{split}/{cls}").mkdir(parents=True, exist_ok=True)

def split_and_copy(src_dir, label):
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir,f))]
    random.shuffle(files)
    n = len(files)
    i = 0
    for split, frac in SPLITS.items():
        take = int(frac * n)
        for f in files[i:i+take]:
            src = os.path.join(src_dir, f)
            dst = os.path.join(DST_ROOT, split, label, f)
            shutil.copy2(src, dst)
        i += take
    # leftover -> train
    for f in files[i:]:
        shutil.copy2(os.path.join(src_dir,f), os.path.join(DST_ROOT,"train",label,f))

if __name__ == "__main__":
    if not os.path.exists(SRC_REAL):
        raise FileNotFoundError(f"Real source folder not found: {SRC_REAL}")
    if not os.path.exists(SRC_SYNTH):
        raise FileNotFoundError(f"Synthetic source folder not found: {SRC_SYNTH}")
    make_dest()
    print("Splitting/copying REAL images...")
    split_and_copy(SRC_REAL, "real")
    print("Splitting/copying SYNTH images...")
    split_and_copy(SRC_SYNTH, "synth")
    print("✅ Done splitting and copying.")
