#!/usr/bin/env python3
"""
Predict script:
python scripts/predict_text_detector.py --model_dir models/text_fake_detector_roberta \
  --input data/new_texts.csv --text-col text --out results/new_preds.csv
"""
import argparse, os, pandas as pd, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--out", default="results/predictions.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(args.device)
    texts = df[args.text_col].astype(str).tolist()
    preds = []
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i:i+args.batch_size]
            enc = tokenizer(batch, truncation=True, padding=True, return_tensors="pt")
            enc = {k: v.to(args.device) for k,v in enc.items()}
            out = model(**enc)
            logits = out.logits.cpu().numpy()
            p = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
            preds.extend(np.argmax(logits, axis=1).tolist())
            probs.extend(p[:,1].tolist())

    df["pred"] = preds
    df["prob_synth"] = probs
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print("Saved predictions to", args.out)

if __name__ == "__main__":
    main()