import numpy as np
import joblib
import json
from sklearn.metrics import classification_report, roc_auc_score

# ----- Load embeddings, labels, and trained model -----
model_path = "models/iso_images.pkl"
x_test_path = "models/embeddings_test.npy"
y_test_path = "models/labels_test.npy"
class_map_path = "models/class_map.json"

print("[eval_iso_verbose] Loading model and data...")
iso = joblib.load(model_path)
X_test = np.load(x_test_path)
y_test = np.load(y_test_path)

with open(class_map_path, "r") as f:
    class_map = json.load(f)

print(f"Class map: {class_map}")
print(f"Test set shape: {X_test.shape}, Labels shape: {y_test.shape}")

# ----- Predictions -----
print("\nComputing Isolation Forest predictions...")
y_pred_scores = -iso.score_samples(X_test)  # lower = more normal
threshold = np.percentile(y_pred_scores, 99)  # 1% contamination threshold
y_pred = (y_pred_scores > threshold).astype(int)  # 0 = real, 1 = fake

# Infer which label corresponds to real
real_idx = None
for k, v in class_map.items():
    if k.lower() == "real":
        real_idx = v
        break
if real_idx is None:
    real_idx = min(class_map.values())

# ----- Adjust labels if necessary -----
y_true = (y_test != real_idx).astype(int)  # 1 = fake, 0 = real

# ----- Evaluation -----
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["real", "fake"]))

try:
    roc_auc = roc_auc_score(y_true, y_pred_scores)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
except Exception as e:
    print("Could not compute ROC-AUC:", e)

print("\n[eval_iso_verbose] Evaluation complete.")
