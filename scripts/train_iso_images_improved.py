# scripts/train_iso_images_improved.py
"""
Improved IsolationForest pipeline for image embeddings.

Features:
- Loads embeddings from models/embeddings_*.npy and models/labels_*.npy
- Scales features with StandardScaler
- Trains IsolationForest with tuned defaults
- Finds best threshold on validation set (Youden's J / maximize TPR-FPR)
- Evaluates on test set (ROC-AUC + classification report)
- Optionally trains a small logistic regression on iso scores (hybrid)
- Saves iso model and scaler to models/iso_images_improved.pkl and models/iso_scaler_improved.pkl
"""

import os
import json
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression

# Config
EMB_DIR = "models"
EMB_TRAIN = os.path.join(EMB_DIR, "embeddings_train.npy")
LAB_TRAIN = os.path.join(EMB_DIR, "labels_train.npy")
EMB_VAL = os.path.join(EMB_DIR, "embeddings_val.npy")
LAB_VAL = os.path.join(EMB_DIR, "labels_val.npy")
EMB_TEST = os.path.join(EMB_DIR, "embeddings_test.npy")
LAB_TEST = os.path.join(EMB_DIR, "labels_test.npy")
CLASS_MAP = os.path.join(EMB_DIR, "class_map.json")

OUT_ISO = os.path.join(EMB_DIR, "iso_images_improved.pkl")
OUT_SCALER = os.path.join(EMB_DIR, "iso_scaler_improved.pkl")
OUT_HYBRID = os.path.join(EMB_DIR, "iso_hybrid_logreg.pkl")

RANDOM_STATE = 42

# Hyperparameters (good defaults for high-dim embeddings)
ISO_PARAMS = {
    "n_estimators": 400,
    "max_samples": 0.8,     # fraction or int
    "contamination": 0.4,   # treat ~40% as anomalies (tunable)
    "max_features": 0.9,
    "bootstrap": True,
    "random_state": RANDOM_STATE,
    "verbose": 0,
    "n_jobs": 1,
}

DO_HYBRID = True  # if True, train logistic regression on iso scores + optional other features

def load_files():
    for p in (EMB_TRAIN, LAB_TRAIN, EMB_VAL, LAB_VAL, EMB_TEST, LAB_TEST, CLASS_MAP):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")
    X_tr = np.load(EMB_TRAIN)
    y_tr = np.load(LAB_TRAIN)
    X_val = np.load(EMB_VAL)
    y_val = np.load(LAB_VAL)
    X_test = np.load(EMB_TEST)
    y_test = np.load(LAB_TEST)
    with open(CLASS_MAP, "r") as f:
        cmap = json.load(f)
    return X_tr, y_tr, X_val, y_val, X_test, y_test, cmap

def find_real_index(cmap):
    # Prefer explicit 'real'; fallback to min index
    for k,v in cmap.items():
        if k.lower() == "real":
            return v
    # try contains
    for k,v in cmap.items():
        if "real" in k.lower():
            return v
    return min(cmap.values())

def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return scaler, X_tr_s, X_val_s, X_test_s

def train_isolationforest(X_real_train, iso_params):
    iso = IsolationForest(**iso_params)
    iso.fit(X_real_train)
    return iso

def choose_threshold(scores_val, y_val_binary):
    # We want threshold on anomaly score where higher score => more anomalous
    # Use ROC curve thresholds and maximize TPR - FPR (Youden's J)
    fpr, tpr, thr = roc_curve(y_val_binary, scores_val)
    j = tpr - fpr
    best_idx = np.argmax(j)
    best_thr = thr[best_idx]
    return best_thr, fpr, tpr, thr, best_idx

def evaluate(y_true_bin, scores_test, threshold):
    y_pred = (scores_test >= threshold).astype(int)
    try:
        auc = roc_auc_score(y_true_bin, scores_test)
    except Exception:
        auc = None
    pr, rc, f1, _ = precision_recall_fscore_support(y_true_bin, y_pred, average="binary", zero_division=0)
    report = classification_report(y_true_bin, y_pred, digits=4, zero_division=0)
    return auc, pr, rc, f1, report, y_pred

def main():
    print("[*] Loading data...")
    X_tr, y_tr, X_val, y_val, X_test, y_test, cmap = load_files()
    print("[*] Data shapes:", X_tr.shape, y_tr.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
    real_idx = find_real_index(cmap)
    print("[*] class_map:", cmap, " -> real index =", real_idx)

    # Prepare "real" samples for unsupervised training
    X_tr_real = X_tr[y_tr == real_idx]
    print(f"[*] Real samples in train: {X_tr_real.shape[0]}")

    # Scale everything
    scaler, X_tr_s, X_val_s, X_test_s = scale_data(X_tr, X_val, X_test)
    # But we will train iso on scaled real-only
    X_tr_real_s = scaler.transform(X_tr_real)

    # Train IsolationForest
    print("[*] Training IsolationForest with params:", ISO_PARAMS)
    iso = train_isolationforest(X_tr_real_s, ISO_PARAMS)

    # Compute anomaly scores: use -score_samples so higher => more anomalous (same convention used earlier)
    scores_val = -iso.score_samples(X_val_s)
    # Convert y_val to binary (1 = fake/synth, 0 = real)
    y_val_bin = (y_val != real_idx).astype(int)

    # Choose threshold via validation
    best_thr, fpr, tpr, thr_arr, best_idx = choose_threshold(scores_val, y_val_bin)
    print(f"[*] Best threshold (Youden's J) on val: {best_thr:.6f} (index {best_idx})")

    # Evaluate on test
    scores_test = -iso.score_samples(X_test_s)
    y_test_bin = (y_test != real_idx).astype(int)
    auc, pr, rc, f1, report, y_pred_test = evaluate(y_test_bin, scores_test, best_thr)

    print("\n=== IsolationForest Evaluation (threshold from val) ===")
    print("ROC-AUC:", None if auc is None else f"{auc:.4f}")
    print(f"Precision: {pr:.4f}, Recall: {rc:.4f}, F1: {f1:.4f}")
    print("\nClassification report (test, using chosen threshold):")
    print(report)

    # Save model and scaler
    joblib.dump(iso, OUT_ISO)
    joblib.dump(scaler, OUT_SCALER)
    print(f"[*] Saved iso model to {OUT_ISO} and scaler to {OUT_SCALER}")

    # Optional hybrid: train logistic regression on iso score (and optionally other simple features)
    if DO_HYBRID:
        print("[*] Training hybrid logistic regression on iso score (val + train pooled)...")
        # Prepare features: iso score on train+val (we'll build small training set)
        scores_tr = -iso.score_samples(X_tr_s)
        scores_val = -iso.score_samples(X_val_s)
        # Build dataset (stack train+val)
        X_hybrid = np.concatenate([scores_tr.reshape(-1,1), scores_val.reshape(-1,1)], axis=0)
        y_hybrid = np.concatenate([(y_tr != real_idx).astype(int), (y_val != real_idx).astype(int)], axis=0)
        # scale small feature
        from sklearn.preprocessing import StandardScaler as SS2
        s2 = SS2()
        X_hybrid_s = s2.fit_transform(X_hybrid)
        X_test_h = s2.transform(scores_test.reshape(-1,1))
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        lr.fit(X_hybrid_s, y_hybrid)
        y_test_proba = lr.predict_proba(X_test_h)[:,1]
        y_test_pred_h = lr.predict(X_test_h)
        auc_h = roc_auc_score(y_test_bin, y_test_proba)
        pr_h, rc_h, f1_h, _ = precision_recall_fscore_support(y_test_bin, y_test_pred_h, average="binary", zero_division=0)
        print("\n--- Hybrid Logistic Regression results (iso score → LR) ---")
        print("ROC-AUC (hybrid):", f"{auc_h:.4f}")
        print(f"Precision: {pr_h:.4f}, Recall: {rc_h:.4f}, F1: {f1_h:.4f}")
        joblib.dump((lr, s2), OUT_HYBRID)
        print(f"[*] Saved hybrid LR to {OUT_HYBRID}")

    print("[*] Done.")

if __name__ == "__main__":
    main()
