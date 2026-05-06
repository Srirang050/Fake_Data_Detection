"""
Usage:
  python predict.py "This movie was terrible because..."
"""
import sys, os, joblib
import numpy as np
from gatekeeper.features import handcrafted_features, transform_tfidf
from gatekeeper.detectors import embed_texts

MODEL_DIR = "models"
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vect.pkl")
TFIDF_LR = os.path.join(MODEL_DIR, "tfidf_lr.pkl")
ISO_PATH = os.path.join(MODEL_DIR, "iso.pkl")

if len(sys.argv) < 2:
    print("Usage: python predict.py \"text here\"")
    sys.exit(1)

text = sys.argv[1]
# load models
vect = joblib.load(TFIDF_PATH)
clf = joblib.load(TFIDF_LR)
iso = joblib.load(ISO_PATH)

# features
x_tfidf = transform_tfidf(vect, [text])
p_super = clf.predict_proba(x_tfidf)[:,1][0]
emb = embed_texts([text], model_name="all-MiniLM-L6-v2")
iso_score = -iso.decision_function(emb)[0]
# normalize iso score (very rough)
# For single sample, we can't scale properly. Use a heuristic threshold:
risk = 0.7 * p_super + 0.3 * (iso_score / (abs(iso_score) + 1.0))
print(f"supervised_prob={p_super:.4f}, iso_score={iso_score:.4f}, risk_score={risk:.4f}")
print("PRED:", "FAKE" if risk >= 0.5 else "REAL")

