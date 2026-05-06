# scripts/eval_iso.py
import numpy as np
from sklearn.metrics import roc_auc_score
import joblib
import os

EMB_DIR = "models"
ISO_OUT = os.path.join(EMB_DIR, "iso_images.pkl")

def eval_iso():
    X_test = np.load(os.path.join(EMB_DIR, "embeddings_test.npy"))
    y_test = np.load(os.path.join(EMB_DIR, "labels_test.npy"))
    iso = joblib.load(ISO_OUT)
    scores = -iso.decision_function(X_test)  # higher -> more anomalous
    auc = roc_auc_score(y_test, scores)
    print("IsolationForest ROC-AUC (higher is better):", auc)

if __name__ == "__main__":
    eval_iso()