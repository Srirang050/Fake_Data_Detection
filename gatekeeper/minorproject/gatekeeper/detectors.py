"""
Supervised TF-IDF + LogisticRegression trainer and
Unsupervised IsolationForest on sentence embeddings.
"""
import os
from typing import List, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# sentence-transformers for embeddings
def embed_texts(texts: List[str], model_name: str="all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        raise RuntimeError("Install sentence-transformers: pip install sentence-transformers")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

def train_supervised_tfidf(X_tfidf, y, save_path):
    """
    Train a TF-IDF + LogisticRegression pipeline and save it.
    """
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # TFIDF is sparse; use with_mean=False
        ("lr", LogisticRegression(max_iter=2000))
    ])
    clf.fit(X_tfidf, y)
    joblib.dump(clf, save_path)
    return clf

def train_isolation_forest(embeddings: np.ndarray, save_path, contamination=0.01):
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    iso.fit(embeddings)
    joblib.dump(iso, save_path)
    return iso

def load_model(path):
    return joblib.load(path)

