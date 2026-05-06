# week1_text_detector.py
# Full working Week-1 script for detecting synthetic vs real IMDB reviews
# Requirements (inside dataguard venv):
# pip install pandas numpy scikit-learn transformers torch tqdm joblib

import os
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# ------------ Configuration ------------
IMDB_CSV = "IMDB Dataset.csv"         # local CSV file (must exist)
SYN_CSV = "synthetic_reviews.csv"     # cached synthetic reviews
MODEL_OUT = "text_fake_detector_clf.pkl"
SCALER_OUT = "text_fake_detector_scaler.pkl"
SYN_COUNT = 1000                      # synthetic samples to use/generate
REAL_COUNT = 1000                     # real samples to use
PPL_BATCH = 16                        # batch size for perplexity computation
GEN_PROMPT = "This movie was"
GEN_NEW_TOKENS = 50                   # generated token count
# ---------------------------------------

print("Phase 1 text detector starting...")

# ---------- 1) Load real reviews ----------
if not os.path.exists(IMDB_CSV):
    raise FileNotFoundError(f"IMDB CSV file not found at: {IMDB_CSV}")

print("Loading IMDB dataset from CSV...")
df = pd.read_csv(IMDB_CSV)
# Check common column names and pick one
if "review" in df.columns:
    real_texts = list(df["review"].astype(str).tolist()[:REAL_COUNT])
elif "text" in df.columns:
    real_texts = list(df["text"].astype(str).tolist()[:REAL_COUNT])
else:
    # fallback: use the first string column
    text_cols = [c for c in df.columns if df[c].dtype == object]
    if not text_cols:
        raise ValueError("No text column found in CSV.")
    real_texts = list(df[text_cols[0]].astype(str).tolist()[:REAL_COUNT])

print(f"Loaded {len(real_texts)} real reviews.")

# ---------- 2) Load or generate synthetic reviews ----------
use_generation = True
if os.path.exists(SYN_CSV):
    try:
        df_fake = pd.read_csv(SYN_CSV)
        if "synthetic_review" in df_fake.columns and len(df_fake) >= SYN_COUNT:
            synthetic_texts = list(df_fake["synthetic_review"].astype(str).tolist()[:SYN_COUNT])
            use_generation = False
            print(f"Loaded {len(synthetic_texts)} synthetic reviews from {SYN_CSV}.")
        else:
            print(f"{SYN_CSV} exists but doesn't contain enough samples; will generate new ones.")
    except Exception as e:
        print("Could not read synthetic CSV; will generate. Error:", e)

if use_generation:
    print("Generating synthetic reviews with GPT-2 (this can take time)...")
    # delayed import so script fails earlier if no transformers/torch installed
    from transformers import pipeline
    generator = pipeline("text-generation", model="gpt2", device=-1)  # CPU device
    synthetic_texts = []
    for _ in tqdm(range(SYN_COUNT), desc="Generating"):
        out = generator(GEN_PROMPT, max_new_tokens=GEN_NEW_TOKENS, num_return_sequences=1, truncation=True)
        synthetic_texts.append(out[0]["generated_text"])
    pd.DataFrame({"synthetic_review": synthetic_texts}).to_csv(SYN_CSV, index=False)
    print(f"Saved {len(synthetic_texts)} synthetic reviews to {SYN_CSV}.")

# ---------- 3) Prepare GPT-2 for perplexity ----------
print("Preparing GPT-2 for perplexity computation (CPU). This may be slow.")

try:
    import torch
    from transformers import GPT2TokenizerFast, GPT2LMHeadModel
except Exception as e:
    raise RuntimeError("Install torch and transformers in your venv (pip install torch transformers). Error: " + str(e))

device = torch.device("cpu")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
# Fix padding token for GPT-2 (no pad token by default)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Move model to device
model.to(device)
model.eval()

# ---------- 4) Perplexity helper (batched) ----------
import torch.nn.functional as F  # used for log_softmax

def perplexity_batch(texts, max_length=512):
    """
    Compute per-sample perplexities for a list of texts.
    Uses token-level log-probabilities to compute per-example perplexities.
    """
    perps = []
    with torch.no_grad():
        for i in range(0, len(texts), PPL_BATCH):
            batch = texts[i:i+PPL_BATCH]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)  # type: ignore
            # Compute per-token log-probs and then per-sample ppl
            logits = outputs.logits  # (B, T, V)
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()
            log_probs = F.log_softmax(shift_logits, dim=-1)
            # gather log-probs of true labels
            true_token_logprobs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            masked_token_logprobs = true_token_logprobs * shift_mask
            sum_logprob_per_sample = masked_token_logprobs.sum(dim=1)
            token_count_per_sample = shift_mask.sum(dim=1).clamp(min=1)
            neg_avg_ll = - (sum_logprob_per_sample / token_count_per_sample)
            per_sample_ppl = torch.exp(neg_avg_ll).cpu().numpy().tolist()
            perps.extend(per_sample_ppl)
    return perps

# ---------- 5) Compute perplexities (with progress bars) ----------
print("Computing perplexities for real texts...")
perp_real = []
for i in tqdm(range(0, len(real_texts), PPL_BATCH), desc="Real Perplexity"):
    perp_real.extend(perplexity_batch(real_texts[i:i+PPL_BATCH]))

print("Computing perplexities for synthetic texts...")
perp_synth = []
for i in tqdm(range(0, len(synthetic_texts), PPL_BATCH), desc="Synthetic Perplexity"):
    perp_synth.extend(perplexity_batch(synthetic_texts[i:i+PPL_BATCH]))

# Trim to length (safety)
perp_real = perp_real[:len(real_texts)]
perp_synth = perp_synth[:len(synthetic_texts)]
print(f"Perplexities computed: real={len(perp_real)}, synthetic={len(perp_synth)}")

# ---------- 6) Feature extraction (include perplexity) ----------
def extract_features_with_ppl(text, ppl_value):
    words = text.split()
    num_words = len(words)
    unique_words = len(set(words))
    avg_sentence_length = np.mean([len(s.split()) for s in text.split('.') if s]) if text.strip() else 0.0
    repetition_score = num_words / (unique_words + 1)
    return [num_words, unique_words, avg_sentence_length, repetition_score, float(ppl_value)]

print("Extracting features and building dataset...")
X_real = [extract_features_with_ppl(t, p) for t, p in zip(real_texts, perp_real)]
X_synth = [extract_features_with_ppl(t, p) for t, p in zip(synthetic_texts, perp_synth)]

X = np.array(X_real + X_synth)
y = np.array([0]*len(X_real) + [1]*len(X_synth))

# ---------- 7) Scale, train, evaluate ----------
print("Scaling features and training classifier...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
try:
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.4f}")
except Exception:
    pass

# ---------- 8) Save model & scaler ----------
joblib.dump(clf, MODEL_OUT)
joblib.dump(scaler, SCALER_OUT)
print(f"\nSaved classifier to {MODEL_OUT} and scaler to {SCALER_OUT}")

print("\nDone. Next runs will reuse synthetic_reviews.csv and the saved models for faster iteration.")
