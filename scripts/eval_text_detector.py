#!/usr/bin/env python3
"""
Evaluate saved model on a CSV and print confusion matrix + top misclassified examples.
Usage:
python scripts/eval_text_detector.py --model_dir models/text_fake_detector_roberta \
  --input data/combined_texts.csv --text-col text --label-col label --out results/eval.csv
"""
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_data(path, text_col="text", label_col="label"):
    df = pd.read_csv(path)
    return df

def predict_batch(model, tokenizer, texts, device="cpu", batch_size=32, max_length=256):
    model.to(device)
    model.eval()
    preds = []
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k,v in enc.items()}
            out = model(**enc)
            logits = out.logits.cpu().numpy()
            p = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
            preds.extend(np.argmax(logits, axis=1).tolist())
            probs.extend(p[:,1].tolist())
    return np.array(preds), np.array(probs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    df = load_data(args.input, args.text_col, args.label_col)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    preds, probs = predict_batch(model, tokenizer, df[args.text_col].tolist(), device=args.device)

    print("Classification report:")
    print(classification_report(df[args.label_col].astype(int), preds, digits=4))
    cm = confusion_matrix(df[args.label_col].astype(int), preds)
    print("Confusion matrix:\n", cm)

    df["pred"] = preds
    df["prob_synth"] = probs

    # Show top misclassified (by probability confidence)
    mis = df[df["pred"] != df[args.label_col].astype(int)].copy()
    if not mis.empty:
        mis["conf"] = np.abs(mis["prob_synth"] - 0.5)
        mis = mis.sort_values("conf", ascending=False)
        print("Top 10 misclassified examples (text, label, pred, prob_synth):")
        for idx, row in mis.head(10).iterrows():
            print(f"\nIndex {idx} label={row[args.label_col]} pred={row['pred']} prob_synth={row['prob_synth']:.4f}\n{row[args.text_col][:400]}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df.to_csv(args.out, index=False)
        print("Saved predictions to", args.out)

if __name__ == "__main__":
    main()
