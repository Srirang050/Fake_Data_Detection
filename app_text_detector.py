# app_text_detector.py
"""
Streamlit UI for text real vs synthetic detector.

Usage:
  conda activate dataguard-embed
  pip install -r requirements.txt
  streamlit run app_text_detector.py

Expectations:
- Your trained huggingface model + tokenizer are in a folder, e.g.
  models/text_fake_detector_roberta_dedup
"""
from pathlib import Path
import io
import tempfile
import time
import math
from typing import List, Tuple

import pandas as pd
import numpy as np
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
from tqdm import tqdm

# ---------- Config ----------
DEFAULT_MODEL_DIR = "models/text_fake_detector_roberta_dedup"  # change if needed
TEXT_COLUMN_DEFAULT = "text"
LABEL_MAP = {0: "real", 1: "synth"}
BATCH_SIZE_DEFAULT = 32
MAX_LENGTH_DEFAULT = 256

# ---------- Helpers ----------
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_dir: str):
    """Load tokenizer + model once. Returns (tokenizer, model, device)"""
    # device selection: prefer GPU (cuda) > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            # MPS (Apple silicon) support
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        except Exception:
            device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model, device

def infer_texts(tokenizer, model, device, texts: List[str], batch_size:int=32, max_length:int=256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (pred_labels (int array), pred_probs_for_label1 (float array))
    label1 is 'synth' in our mapping.
    """
    all_preds = []
    all_probs1 = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            logits = out.logits.cpu()
            probs = softmax(logits, dim=1).numpy()  # shape (B, 2)
            preds = np.argmax(logits.numpy(), axis=1)
            all_preds.extend(preds.tolist())
            all_probs1.extend(probs[:,1].tolist())

    return np.array(all_preds, dtype=int), np.array(all_probs1, dtype=float)

def safe_read_uploaded_file(uploaded_file: io.BytesIO, text_col_hint=TEXT_COLUMN_DEFAULT):
    """Try CSV/TSV/JSON auto-detect and return DataFrame"""
    # attempt common formats
    uploaded_file.seek(0)
    name = getattr(uploaded_file, "name", "")
    try:
        if name.endswith(".csv") or name.endswith(".txt") or name.endswith(".tsv"):
            # try comma first
            try:
                df = pd.read_csv(uploaded_file)
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, delimiter="\t")
        elif name.endswith(".json") or name.endswith(".jsonl"):
            df = pd.read_json(uploaded_file, lines=True)
        else:
            # fallback: try CSV then JSONL
            try:
                df = pd.read_csv(uploaded_file)
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_json(uploaded_file, lines=True)
    except Exception as e:
        raise RuntimeError(f"Could not parse uploaded file: {e}")
    if df.empty:
        raise RuntimeError("Uploaded file parsed but is empty.")
    return df

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Text Real vs Synthetic Detector", layout="wide")
st.title("Text Real vs Synthetic Detector — Upload dataset and get percentages")

st.sidebar.header("Model & settings")
model_dir = st.sidebar.text_input("Model directory", value=DEFAULT_MODEL_DIR)
batch_size = st.sidebar.number_input("Batch size (inference)", min_value=1, max_value=512, value=BATCH_SIZE_DEFAULT)
max_length = st.sidebar.number_input("Max token length", min_value=32, max_value=1024, value=MAX_LENGTH_DEFAULT)
show_examples = st.sidebar.checkbox("Show example predictions (first 10 rows)", value=True)
trust_threshold = st.sidebar.slider("Confidence threshold to count as 'synth' (probability of label=1)", 0.0, 1.0, 0.5)

st.info("How it works: upload a CSV/TSV/JSONL with a text column (default column name: 'text'). The model will predict whether each row is real (label 0) or synthetic (label 1).")

# load model
with st.spinner("Loading model..."):
    try:
        tokenizer, model, device = load_model_and_tokenizer(model_dir)
        st.sidebar.success(f"Model loaded from {model_dir} (device: {device})")
    except Exception as e:
        st.sidebar.error(f"Failed to load model from {model_dir}: {e}")
        st.stop()

uploaded = st.file_uploader("Upload dataset (CSV / TSV / JSONL). Must include text column.", type=["csv","tsv","txt","json","jsonl"])
if uploaded is not None:
    try:
        df = safe_read_uploaded_file(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    st.write("Columns detected:", list(df.columns))
    text_col = st.selectbox("Choose text column", options=list(df.columns), index=list(df.columns).index(TEXT_COLUMN_DEFAULT) if TEXT_COLUMN_DEFAULT in df.columns else 0)
    st.write("Preview of uploaded data (first 5 rows):")
    st.dataframe(df.head(5))

    # optional pre-processing: drop missing
    if st.checkbox("Drop rows with empty text", value=True):
        before = len(df)
        df = df[ df[text_col].notnull() & (df[text_col].astype(str).str.strip() != "") ].reset_index(drop=True)
        st.write(f"Dropped {before - len(df)} empty rows, {len(df)} remain.")

    if len(df) == 0:
        st.warning("No rows left after cleaning.")
        st.stop()

    run_button = st.button("Run detection on uploaded dataset")
    if run_button:
        start = time.time()
        texts = df[text_col].astype(str).tolist()
        progress = st.progress(0)
        # run in batches but update progress
        all_preds = []
        all_probs = []
        total = len(texts)
        for i in range(0, total, batch_size):
            batch = texts[i:i+batch_size]
            preds, probs1 = infer_texts(tokenizer, model, device, batch, batch_size=batch_size, max_length=max_length)
            all_preds.extend(preds.tolist())
            all_probs.extend(probs1.tolist())
            progress.progress(min(1.0, (i + len(batch)) / total))

        elapsed = time.time() - start
        st.success(f"Inference complete — {total} rows, time: {elapsed:.1f}s")

        # attach results
        df_out = df.copy()
        df_out["_pred_label"] = all_preds
        df_out["_pred_prob_synth"] = all_probs
        # apply optional confidence threshold: a row is 'synth' if prob >= trust_threshold
        df_out["_pred_final"] = df_out["_pred_prob_synth"].apply(lambda p: 1 if p >= trust_threshold else 0)

        # summary counts
        counts = df_out["_pred_final"].value_counts().to_dict()
        count_real = int(counts.get(0, 0))
        count_synth = int(counts.get(1, 0))
        pct_real = 100.0 * count_real / total
        pct_synth = 100.0 * count_synth / total

        # UI summary
        col1, col2 = st.columns([2,3])
        with col1:
            st.metric("Total rows", total)
            st.metric("Predicted real (count)", f"{count_real}", delta=f"{pct_real:.2f}%")
            st.metric("Predicted synthetic (count)", f"{count_synth}", delta=f"{pct_synth:.2f}%")
        with col2:
            # pie chart using pandas
            pie_df = pd.DataFrame({
                "label": ["real", "synth"],
                "count": [count_real, count_synth]
            })
            st.altair_chart(
                (st.altair_chart if False else None)  # placeholder to keep linters quiet
            , use_container_width=True)
            # fallback simple bar & text table
            st.write("Distribution:")
            st.write(pie_df.set_index("label"))

        # show example predictions
        if show_examples:
            st.write("Example predictions (first 20 rows):")
            preview = df_out[[text_col, "_pred_label", "_pred_prob_synth", "_pred_final"]].head(20)
            preview["_pred_label_name"] = preview["_pred_label"].map(LABEL_MAP)
            preview["_pred_final_name"] = preview["_pred_final"].map(LABEL_MAP)
            st.dataframe(preview)

        # download link
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

        # small advice
        st.info("Tip: If model predicts nearly all rows as one class, you may need more varied training data or to inspect dataset for leaks/duplicates.")

else:
    st.write("Upload a dataset to get started.")
    st.write("If you don't have a model yet, you can still try with a small sample: create a CSV with 'text' column and upload.")
