# app_text_detector_fixed.py
import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import altair as alt
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------
# Config / Utilities
# ---------------------
st.set_page_config(page_title="Text Real vs Synthetic Detector", layout="wide")

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_dir_or_name="models/text_fake_detector_roberta", device_str=None):
    """
    Load tokenizer + model once and cache. Accept a path or HF model id.
    """
    device = torch.device(device_str) if device_str else (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    # try local dir first; fallback to model id
    tokenizer = AutoTokenizer.from_pretrained(model_dir_or_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir_or_name, num_labels=2)
    model.to(device)
    model.eval()
    return tokenizer, model, device

def batched_predict_texts(texts, tokenizer, model, device,
                          batch_size=64, max_length=256, progress_callback=None, fp16=False):
    """
    Batch tokenization + inference for lists of strings.
    Returns numpy arrays: preds (0/1) and probs (prob for class 1).
    progress_callback(i, total) gets called per-batch if provided.
    """
    n = len(texts)
    preds = np.zeros(n, dtype=int)
    probs = np.zeros(n, dtype=float)

    # Use torch.no_grad and move batches to device
    dtype = torch.float16 if (fp16 and device.type != "cpu") else torch.float32
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            batch_texts = list(texts[start:end])
            enc = tokenizer(batch_texts,
                            truncation=True,
                            padding=True,
                            max_length=max_length,
                            return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            if "token_type_ids" in enc:
                token_type_ids = enc["token_type_ids"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            if dtype == torch.float16:
                logits = logits.half()
            batch_probs = torch.nn.functional.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            batch_preds = logits.argmax(dim=1).detach().cpu().numpy()
            preds[start:end] = batch_preds
            probs[start:end] = batch_probs
            if progress_callback:
                progress_callback(end, n)
    return preds, probs

def safe_altair_bar(counts_df, title="Distribution"):
    """
    Build an Altair bar chart from a small DataFrame with columns ['label','count','pct'].
    Returns None if input invalid.
    """
    if counts_df is None or len(counts_df) == 0:
        return None
    chart = alt.Chart(counts_df).mark_bar().encode(
        x=alt.X("label:N", title="Class (0 = real, 1 = synthetic)"),
        y=alt.Y("count:Q", title="Count"),
        tooltip=["label", "count", alt.Tooltip("pct:Q", format=".2f")]
    ).properties(width=400, height=300, title=title)
    return chart

# ---------------------
# UI
# ---------------------
st.title("Text Real vs Synthetic Detector — Upload dataset and get percentages")

st.info("Upload a CSV/TSV/JSONL file with a text column. Default text column name is 'text'.")

uploaded = st.file_uploader("Upload dataset (CSV / TSV / JSONL). Must include text column.", type=["csv","tsv","txt","json","jsonl"], accept_multiple_files=False)
col1, col2 = st.columns([2,1])

if uploaded is None:
    st.caption("No file uploaded yet. Use the uploader above.")
    st.stop()

# Read file robustly
upload_bytes = uploaded.read()
try:
    # Try csv first (comma). If .tsv fallback when extension suggests.
    suffix = Path(uploaded.name).suffix.lower()
    if suffix in [".tsv", ".txt"]:
        df = pd.read_csv(io.BytesIO(upload_bytes), sep="\t", dtype=str, encoding="utf-8", engine="python")
    elif suffix in [".json", ".jsonl"]:
        df = pd.read_json(io.BytesIO(upload_bytes), lines=(suffix==".jsonl"))
    else:
        # csv (default)
        df = pd.read_csv(io.BytesIO(upload_bytes), dtype=str, encoding="utf-8", engine="python")
except Exception as e:
    # fallback with latin-1 if utf-8 errors
    try:
        df = pd.read_csv(io.BytesIO(upload_bytes), dtype=str, encoding="latin-1", engine="python")
    except Exception as e2:
        st.error(f"Failed to read uploaded file: {e}\nFallback error: {e2}")
        st.stop()

st.write(f"Columns detected: {list(df.columns)}")
text_col = st.selectbox("Choose text column", options=list(df.columns), index=0)
st.write("Preview of uploaded data (first 5 rows):")
st.dataframe(df.head(5))

# drop rows with missing text
initial_len = len(df)
df[text_col] = df[text_col].astype(str)
df = df[df[text_col].str.strip().astype(bool)].reset_index(drop=True)
dropped = initial_len - len(df)
if dropped > 0:
    st.warning(f"Dropped {dropped} empty rows, {len(df)} remain.")

if len(df) == 0:
    st.error("No valid rows to predict after dropping empty text.")
    st.stop()

# model selection
with col2:
    st.write("Model / inference settings")
    model_path = st.text_input("Model directory or HF model id", value="models/text_fake_detector_roberta")
    batch_size = st.number_input("Batch size", min_value=8, max_value=1024, value=64, step=8)
    max_length = st.number_input("Max tokens", min_value=32, max_value=1024, value=256, step=32)
    fp16 = st.checkbox("Use fp16 (only if GPU/MPS supports)", value=False)
    run_btn = st.button("Run inference")

if not run_btn:
    st.stop()

# load model (cached)
with st.spinner("Loading model..."):
    try:
        tokenizer, model, device = load_model_and_tokenizer(model_path)
    except Exception as e:
        st.error(f"Failed to load model/tokenizer from '{model_path}': {e}")
        st.stop()

st.success(f"Model loaded. Using device: {device}")

# prepare texts
texts = df[text_col].astype(str).tolist()

# progress bar
progress = st.progress(0)
status_text = st.empty()

t0 = time.time()
def _progress_callback(done, total):
    pct = done/total
    progress.progress(int(pct*100))
    status_text.text(f"Inference: {done}/{total} rows")

# run batched inference
with st.spinner("Running batched inference..."):
    try:
        preds, probs = batched_predict_texts(
            texts, tokenizer, model, device,
            batch_size=int(batch_size), max_length=int(max_length),
            progress_callback=_progress_callback, fp16=fp16
        )
    except Exception as e:
        st.error(f"Inference failed: {e}")
        st.stop()
t1 = time.time()
elapsed = t1 - t0
progress.empty()
status_text.text("Done")

# attach results
df_out = df.copy()
df_out["pred"] = preds
df_out["prob_synthetic"] = probs  # prob for class 1 (synthetic)
counts = df_out["pred"].value_counts().sort_index()
total = len(df_out)
counts_dict = {"label": [], "count": [], "pct": []}
for label in [0,1]:
    c = int(counts.get(label, 0))
    counts_dict["label"].append(str(label))
    counts_dict["count"].append(c)
    counts_dict["pct"].append(100.0 * c / total if total>0 else 0.0)
counts_df = pd.DataFrame(counts_dict)

# Display summary
colA, colB, colC = st.columns([1,2,1])
with colA:
    st.metric("Total rows", f"{total}")
    st.metric("Predicted real (count)", f"{counts_dict['count'][0]}", f"{counts_dict['pct'][0]:.2f}%")
with colB:
    chart = safe_altair_bar(counts_df, title="Predicted Real vs Synthetic")
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No chart to display.")
with colC:
    st.metric("Predicted synthetic (count)", f"{counts_dict['count'][1]}", f"{counts_dict['pct'][1]:.2f}%")
    st.write(f"Inference time: {elapsed:.1f}s  (batch_size={batch_size})")

# show top examples
st.subheader("Sample predictions")
st.dataframe(df_out[[text_col, "pred", "prob_synthetic"]].head(50))

# allow download
csv_bytes = df_out.to_csv(index=False).encode("utf-8")
st.download_button("Download predictions CSV", csv_bytes, file_name="predictions.csv", mime="text/csv")

st.success("Done — predictions ready.")
