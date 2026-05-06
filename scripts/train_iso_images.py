# scripts/train_iso_images.py
"""
Train an IsolationForest anomaly detector on image embeddings belonging to the REAL class.
This script is robust to whatever class->index mapping ImageFolder produced; it reads
models/class_map.json to find which index corresponds to the 'real' folder.
"""
import os
import json
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import sys

EMB_DIR = "models"
EMB_X = os.path.join(EMB_DIR, "embeddings_train.npy")
EMB_Y = os.path.join(EMB_DIR, "labels_train.npy")
CLASS_MAP = os.path.join(EMB_DIR, "class_map.json")
ISO_OUT = os.path.join(EMB_DIR, "iso_images.pkl")

def load_class_map(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"class_map.json not found at {path}. Run the dataloader test to create it.")
    with open(path, "r") as f:
        cmap = json.load(f)
    # normalize keys to lowercase for flexible matching
    cmap_lower = {k.lower(): v for k, v in cmap.items()}
    return cmap, cmap_lower

def find_real_index(cmap, cmap_lower):
    # prefer exact 'real' key, else search case-insensitively
    if "real" in cmap:
        return cmap["real"]
    if "real" in cmap_lower:
        # find original key that matched lower-case
        for k, v in cmap.items():
            if k.lower() == "real":
                return v
    # if not found, try common alternatives
    for alt in ("genuine", "authentic"):
        if alt in cmap_lower:
            print(f"Found alternative real-class name '{alt}' in class_map.json, using it.")
            return cmap_lower[alt]
    # not found
    raise ValueError("Could not find a 'real' class name in class_map.json. Please check your training folders and class_map.json.")

def main(contamination=0.01, n_estimators=200):
    # sanity checks
    if not os.path.exists(EMB_X) or not os.path.exists(EMB_Y):
        raise FileNotFoundError(f"Embeddings or labels not found. Expected:\n {EMB_X}\n {EMB_Y}\nRun extract_embeddings.py first.")
    print("Loading embeddings and labels...")
    X = np.load(EMB_X)
    y = np.load(EMB_Y)
    print(f"Embeddings shape: {X.shape}, labels shape: {y.shape}")

    # load class map
    cmap, cmap_lower = load_class_map(CLASS_MAP)
    print("Loaded class map:", cmap)

    real_idx = None
    try:
        real_idx = find_real_index(cmap, cmap_lower)
    except Exception as e:
        print("ERROR locating 'real' class index in class_map.json:", e)
        sys.exit(1)

    print(f"Using real class index = {real_idx} to select normal samples.")

    # select real embeddings
    X_real = X[y == real_idx]
    if X_real.shape[0] == 0:
        raise ValueError(f"No samples found for 'real' class (index {real_idx}). Check class_map.json and labels array.")
    print(f"Number of real (normal) samples for IsolationForest training: {X_real.shape[0]}")

    # train IsolationForest
    print(f"Training IsolationForest (n_estimators={n_estimators}, contamination={contamination}) ...")
    iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42, verbose=0)
    iso.fit(X_real)

    # save model
    os.makedirs(EMB_DIR, exist_ok=True)
    joblib.dump(iso, ISO_OUT)
    print(f"Saved IsolationForest to: {ISO_OUT}")

if __name__ == "__main__":
    # You can edit contamination/n_estimators here if needed
    main(contamination=0.01, n_estimators=200)
