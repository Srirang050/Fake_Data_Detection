# scripts/clean_and_split.py
"""
Usage:
python scripts/clean_and_split.py --input data/combined_texts.csv --out_dir data/splits --dedup_near 0.9
"""
import argparse, os, re
import pandas as pd
from sklearn.model_selection import train_test_split
from difflib import SequenceMatcher

def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    # remove common labels/markers that might have leaked in
    s = re.sub(r"(generated_by|generated:|source:|__synth__|__SYNTH__|<generated>).*", "", s, flags=re.I)
    # remove html tags
    s = re.sub(r"<[^>]+>", " ", s)
    # remove URLs/emails
    s = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", " ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def near_dedupe(df, threshold=0.95):
    # naive O(n^2) approach -> OK for few thousand rows; skip if dataset large
    keep = []
    texts = df['text'].tolist()
    for i, t in enumerate(texts):
        too_similar = False
        for j in keep:
            sim = SequenceMatcher(None, t, texts[j]).ratio()
            if sim >= threshold:
                too_similar = True
                break
        if not too_similar:
            keep.append(i)
    return df.iloc[keep].reset_index(drop=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out_dir", default="data/splits")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--dedup-near", type=float, default=0.0, help="0 to skip, else similarity threshold 0-1")
    args = p.parse_args()

    df = pd.read_csv(args.input, dtype={args.text_col: str, args.label_col: int})
    print("Loaded:", len(df), "rows")
    # clean
    df['text'] = df[args.text_col].astype(str).map(clean_text)
    # drop empty text rows
    df = df[df['text'].str.strip().astype(bool)].reset_index(drop=True)
    # exact dedupe
    before = len(df)
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    print(f"Dropped exact duplicates: {before - len(df)} rows -> {len(df)} remain")
    # near dedupe optional
    if args.dedup_near and 0.0 < args.dedup_near < 1.0:
        print("Running near-duplicate cleanup (this may be slow for large datasets)...")
        before = len(df)
        df = near_dedupe(df, threshold=args.dedup_near)
        print(f"Dropped near-duplicates: {before - len(df)} -> {len(df)} remain")
    # split stratified
    os.makedirs(args.out_dir, exist_ok=True)
    df = df[[ 'text', args.label_col]] if 'text' in df.columns else df
    df = df.rename(columns={args.label_col: 'label'})
    train_df, temp_df = train_test_split(df, test_size=(args.test_size + args.val_size),
                                         stratify=df['label'], random_state=args.seed)
    val_size_rel = args.val_size / (args.val_size + args.test_size)
    val_df, test_df = train_test_split(temp_df, test_size=(1.0 - val_size_rel),
                                       stratify=temp_df['label'], random_state=args.seed)
    train_df.to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.out_dir, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir, "test.csv"), index=False)
    print("Saved splits:", {k: len(v) for k,v in [('train',train_df),('val',val_df),('test',test_df)]})

if __name__ == "__main__":
    main()
