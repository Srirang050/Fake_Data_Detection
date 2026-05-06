"""
feature extraction helpers:
- text_tfidf: returns sparse TF-IDF vectors (sklearn TfidfVectorizer)
- sentence embeddings via sentence-transformers (returns numpy arrays)
- small handcrafted features (word count, unique words, avg sentence len, repetition)
"""
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def handcrafted_features(texts: List[str]):
    X = []
    for t in texts:
        words = t.split()
        n = len(words)
        unique = len(set(words))
        avg_sent = np.mean([len(s.split()) for s in t.split('.') if s]) if t.strip() else 0.0
        repetition = n / (unique + 1)
        X.append([n, unique, avg_sent, repetition])
    return np.array(X, dtype=float)

def build_tfidf(texts: List[str], max_features:int=5000):
    vect = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    X = vect.fit_transform(texts)
    return vect, X

def transform_tfidf(vect: TfidfVectorizer, texts: List[str]):
    return vect.transform(texts)

