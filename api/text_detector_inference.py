# api/text_detector_inference.py
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict

class TextDetector:
    def __init__(self, model_dir: str = "models/text_detector", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model.eval()

    def predict(self, texts: List[str], max_length: int = 256) -> List[Dict]:
        enc = self.tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
            logits = out.logits.detach().cpu().numpy()
        probs = (np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True))[:,1]
        preds = (logits.argmax(axis=1)).tolist()
        results = []
        for p, pr in zip(preds, probs):
            results.append({"label": "synth" if int(p)==1 else "real", "probability_synth": float(pr)})
        return results

# quick usage:
# detector = TextDetector("models/text_detector")
# print(detector.predict(["This is a test sentence."]))
