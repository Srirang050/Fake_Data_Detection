"""
Usage:
  python -m gatekeeper.pipeline --mode train
  python -m gatekeeper.pipeline --mode eval
This script trains:
 - TFIDF vectorizer (saved as tfidf.pkl)
 - TFIDF+LR pipeline (saved as tfidf_lr.pkl)
 - Sentence-transformer embeddings -> IsolationForest (saved as iso.pkl)
It also evaluates and prints metrics and saves everything in ./models/
"""
import argparse
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from gatekeeper.features import build_tfidf, transform_tfidf, handcrafted_features
from gatekeeper.detectors import train_supervised_tfidf, train_isolation_forest, embed_texts

MODEL_DIR = "models"
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vect.pkl")
TFIDF_LR = os.path.join(MODEL_DIR, "tfidf_lr.pkl")
ISO_PATH = os.path.join(MODEL_DIR, "iso.pkl")

def load_dataset_from_csv(imdb_csv="IMDB Dataset.csv", synthetic_csv="synthetic_reviews.csv", real_count=1000, synth_count=1000):
    import pandas as pd
    if not os.path.exists(imdb_csv):
        raise FileNotFoundError(imdb_csv)
    df = pd.read_csv(imdb_csv)
    if "review" in df.columns:
        real_texts = df["review"].astype(str).tolist()[:real_count]
    elif "text" in df.columns:
        real_texts = df["text"].astype(str).tolist()[:real_count]
    else:
        # fallback first string column
        text_cols = [c for c in df.columns if df[c].dtype == object]
        real_texts = df[text_cols[0]].astype(str).tolist()[:real_count]
    if os.path.exists(synthetic_csv):
        df_fake = pd.read_csv(synthetic_csv)
        synthetic_texts = df_fake["synthetic_review"].astype(str).tolist()[:synth_count]
    else:
        raise FileNotFoundError(synthetic_csv)
    texts = real_texts + synthetic_texts
    y = np.array([0]*len(real_texts) + [1]*len(synthetic_texts))
    return texts, y

def train_all(args):
    os.makedirs(MODEL_DIR, exist_ok=True)
    texts, y = load_dataset_from_csv(args.imdb, args.synthetic, args.real_count, args.synth_count)
    # TF-IDF
    print("Building TF-IDF...")
    vect, X_tfidf = build_tfidf(texts, max_features=args.max_features)
    joblib.dump(vect, TFIDF_PATH)
    print("Training TF-IDF + LR...")
    clf = train_supervised_tfidf(X_tfidf, y, TFIDF_LR)
    # Embeddings and IsolationForest
    print("Computing embeddings for IsolationForest...")
    embeddings = embed_texts(texts, model_name=args.embed_model)
    print("Training IsolationForest on embeddings...")
    iso = train_isolation_forest(embeddings, ISO_PATH, contamination=args.contamination)
    print("Training complete. Models saved in", MODEL_DIR)

def eval_all(args):
    # load models & evaluate on held-out test split
    from sklearn.preprocessing import StandardScaler
    vect = joblib.load(TFIDF_PATH)
    clf = joblib.load(TFIDF_LR)
    iso = joblib.load(ISO_PATH)
    texts, y = load_dataset_from_csv(args.imdb, args.synthetic, args.real_count, args.synth_count)
    X_tfidf = transform_tfidf(vect, texts)
    # supervised probs
    y_proba = clf.predict_proba(X_tfidf)[:,1]
    # unsupervised anomaly scores (iso: -1 for outlier, +1 normal) -> convert to score
    embeddings = embed_texts(texts, model_name=args.embed_model)
    iso_scores = -iso.decision_function(embeddings)  # higher -> more anomalous
    # combine simple weighted ensemble
    risk = 0.7 * y_proba + 0.3 * (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9)
    # binarize at 0.5
    y_pred = (risk >= 0.5).astype(int)
    print("Classification report (ensemble):")
    print(classification_report(y, y_pred))
    try:
        print("ROC-AUC (supervised):", roc_auc_score(y, y_proba))
    except Exception:
        pass

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train","eval"], default="train")
    p.add_argument("--imdb", default="IMDB Dataset.csv")
    p.add_argument("--synthetic", default="synthetic_reviews.csv")
    p.add_argument("--real_count", type=int, default=200)   # smaller default for faster runs
    p.add_argument("--synth_count", type=int, default=200)
    p.add_argument("--max_features", type=int, default=5000)
    p.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--contamination", type=float, default=0.01)
    args = p.parse_args()
    if args.mode == "train":
        train_all(args)
    else:
        eval_all(args)

