#!/usr/bin/env python3
"""
near_dedupe_fast.py

Fast near-duplicate removal using sentence-transformers embeddings + sklearn NearestNeighbors.

Usage:
python scripts/near_dedupe_fast.py --input data/combined_texts.csv --out data/combined_texts_dedup_fast.csv --text-col text --threshold 0.92
"""
import argparse, os
import pandas as pd
import numpy as np
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--threshold", type=float, default=0.92, help="cosine similarity threshold (0..1)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="sentence-transformers model")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype=str, low_memory=False)
    if args.text_col not in df.columns:
        raise SystemExit(f"Column '{args.text_col}' not found. Columns: {list(df.columns)}")

    texts = df[args.text_col].fillna("").astype(str).tolist()
    n = len(texts)
    print(f"Loaded {n} rows")

    # lazy import heavy libs so script fails early if missing
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise SystemExit("Install sentence-transformers (pip install sentence-transformers) to use this script. " + str(e))
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception as e:
        raise SystemExit("Install scikit-learn (pip install scikit-learn) to use this script. " + str(e))

    model = SentenceTransformer(args.model)
    print("Computing embeddings (will use CPU; batch-size=%d)..." % args.batch_size)
    embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    # Use cosine similarity via nearest neighbors with metric='cosine' (scikit returns distances = 1 - cosine_similarity)
    neigh = NearestNeighbors(n_neighbors=10, metric="cosine", n_jobs=-1)  # we'll query a few neighbors
    neigh.fit(embeddings)

    # For each point, find neighbors within threshold: distance <= (1 - threshold)
    sim_threshold = args.threshold
    dist_threshold = 1.0 - sim_threshold
    to_keep = np.ones(n, dtype=bool)
    # We'll iterate in order and mark near duplicates as removed (keep first occurrence)
    for i in range(n):
        if not to_keep[i]:
            continue
        # query a few nearest neighbors (some may be further than threshold)
        dists, idxs = neigh.kneighbors(embeddings[i].reshape(1, -1), n_neighbors=50, return_distance=True)
        dists = dists.flatten()
        idxs = idxs.flatten()
        # for neighbors with distance <= dist_threshold and index > i, mark as duplicate
        close_mask = (dists <= dist_threshold)
        close_idxs = idxs[close_mask]
        for j in close_idxs:
            if j == i:
                continue
            # only remove later entries so we keep earliest
            if j > i:
                to_keep[j] = False

    kept_df = df.loc[to_keep].reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kept_df.to_csv(args.out, index=False)
    print(f"Wrote deduped file: {args.out}  (kept {len(kept_df)} of {n})")

if __name__ == "__main__":
    main()
