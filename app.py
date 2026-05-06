"""
app_detector.py

Streamlit app to upload a text dataset (CSV/TSV/JSONL) and/or an image dataset (zip or folder),
run your text & image 'real vs synthetic' detectors, and show percentages and low-confidence examples.

Assumptions / behavior:
- Text model: tries (in order)
    1) HF Transformers model dir at models/text_fake_detector_roberta (AutoTokenizer + AutoModelForSequenceClassification)
    2) sklearn pipeline saved at models/text_fake_detector_clf.pkl (joblib)
- Image model: expects a ResNet-style checkpoint saved as a dict { "model_state_dict": ..., "classes": [...] }
    at models/image_fake_detector_try1/best_model.pth (or change path in UI).
- If a model isn't found, the app will tell you and allow you to still upload data / preview it.
- Use -- streamlit run app_detector.py
"""
import streamlit as st
import tempfile, os, io, zipfile, shutil, math, time
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models
import joblib
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Real vs Synthetic Detector", page_icon="🔎")

# ---------------------
# Helpers: device
# ---------------------
@st.cache_resource
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch, "has_mps", False) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()

# ---------------------
# Text model loader / predictor
# ---------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource
def load_text_transformer_model(model_dir: str):
    """Try to load a HF Transformers dir. Returns (tokenizer, model) or raises."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_text_sklearn_model(path: str):
    """Load an sklearn pipeline or (vectorizer,clf) saved with joblib."""
    return joblib.load(path)

def predict_texts_transformer(tokenizer, model, texts: List[str], batch_size=64):
    results = []
    model.to(device)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        toks = tokenizer(batch, truncation=True, padding=True, return_tensors="pt", max_length=256)
        toks = {k: v.to(device) for k,v in toks.items()}
        with torch.no_grad():
            out = model(**toks)
            logits = out.logits.cpu()
            probs = torch.nn.functional.softmax(logits, dim=1).numpy()
            preds = probs.argmax(axis=1)
        for j, p in enumerate(preds):
            results.append({"pred": int(p), "prob": float(np.max(probs[j]))})
    return results

def predict_texts_sklearn(pipe, texts: List[str]):
    # sklearn pipeline should have predict_proba or decision_function
    results = []
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(texts)
        preds = probs.argmax(axis=1)
        for j in range(len(preds)):
            results.append({"pred": int(preds[j]), "prob": float(np.max(probs[j]))})
    else:
        preds = pipe.predict(texts)
        # fallback: probability unknown, set 1.0
        for j in range(len(preds)):
            results.append({"pred": int(preds[j]), "prob": 1.0})
    return results

# ---------------------
# Image model loader / predictor
# ---------------------
def build_image_model(num_classes: int = 2, backbone_name="resnet50", pretrained=True):
    if backbone_name.startswith("resnet"):
        # use torchvision resnet50
        if pretrained:
            try:
                weights = getattr(models, "ResNet50_Weights").IMAGENET1K_V1
                model = models.resnet50(weights=weights)
            except Exception:
                model = models.resnet50(weights=None)
        else:
            model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        # fallback
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

def load_image_checkpoint(path: str, num_classes=2):
    state = torch.load(path, map_location="cpu")
    # state may be dict containing 'model_state_dict' (your saved format) or be a direct state_dict
    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
        classes = state.get("classes", None)
    elif isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
        classes = state.get("classes", None)
    else:
        # assume it's a state dict
        state_dict = state
        classes = None
    model = build_image_model(num_classes=num_classes, pretrained=True)
    missing = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, classes

class ImageFolderSimple(Dataset):
    def __init__(self, filepaths: List[str], transform=None):
        self.files = filepaths
        self.transform = transform
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, path

def predict_images(model, filepaths: List[str], img_size=128, batch_size=32):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    ds = ImageFolderSimple(filepaths, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    results = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            imgs, paths = batch
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            for i, p in enumerate(preds):
                results.append({"path": paths[i], "pred": int(p), "prob": float(np.max(probs[i]))})
    return results

# ---------------------
# Utility: unzip / gather images
# ---------------------
def save_uploadedfile_to_tempfile(uploaded_file) -> str:
    t = tempfile.mkdtemp()
    p = os.path.join(t, uploaded_file.name)
    with open(p, "wb") as fh:
        fh.write(uploaded_file.getbuffer())
    return p

def gather_image_paths_from_zip(zip_path: str, outdir: Optional[str] = None) -> List[str]:
    tmp = outdir or tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp)
    # find images recursively
    exts = (".jpg",".jpeg",".png",".bmp",".webp")
    files = []
    for root,_,fnames in os.walk(tmp):
        for fn in fnames:
            if fn.lower().endswith(exts):
                files.append(os.path.join(root,fn))
    return files

# ---------------------
# UI
# ---------------------
st.title("🔎 Real vs Synthetic — Unified Detector (text + image)")

st.markdown("""
Upload either a text dataset (CSV/TSV/JSONL) containing a text column *or* upload an image dataset (zip of images / folder as zip).
The app will run your trained detectors (if present under `models/`) and give counts, percentages and sample low-confidence cases.
""")

col1, col2 = st.columns([1,2])

with col1:
    st.header("Models / Settings")

    # text model path selector
    text_model_dir = st.text_input("Text model dir (Transformers)", value="models/text_fake_detector_roberta")
    text_sklearn_path = st.text_input("Text sklearn model (joblib)", value="models/text_fake_detector_clf.pkl")
    use_text = st.checkbox("Enable text dataset detection", value=True)

    # image model path
    image_model_path = st.text_input("Image model checkpoint (.pth)", value="models/image_fake_detector_try1/best_model.pth")
    enable_image = st.checkbox("Enable image dataset detection", value=True)

    # common options
    confidence_threshold = st.slider("Low-confidence threshold (flag examples with max-prob < ...)", 0.50, 0.95, 0.70)
    img_size = st.selectbox("Image size for inference", [128, 160, 224], index=0)
    batch_size = st.number_input("Batch size for inference", min_value=8, max_value=128, value=32)

with col2:
    st.header("Upload data")
    tabs = st.tabs(["Text dataset", "Image dataset", "Combined (both)"])
    # Text tab
    with tabs[0]:
        uploaded_text = st.file_uploader("Upload CSV / TSV / JSONL (text column)", type=['csv','tsv','json','jsonl'])
        infer_text_button = st.button("Run text inference", key="run_text")
        text_preview_n = st.number_input("Preview rows", min_value=1, max_value=20, value=5)

    # Image tab
    with tabs[1]:
        uploaded_images = st.file_uploader("Upload images ZIP (folders inside allowed) or individual images (zip preferred)", type=['zip','tar','gz','tgz'], accept_multiple_files=False)
        # allow user to also point to folder on disk (if running locally)
        local_image_folder = st.text_input("Local image folder (optional, e.g. data/images/test)", "")
        infer_image_button = st.button("Run image inference", key="run_img")

    # Combined tab: both uploaders present already above
    with tabs[2]:
        st.info("Use the previous tabs to upload text / image datasets and run inferences. Combined percentages will be shown automatically when results are available.")

# ---------------------
# Try loading models (non-blocking)
# ---------------------
text_tokenizer, text_transformer_model, sklearn_pipe, image_model = None, None, None, None
text_transformer_available = False
text_sklearn_available = False
image_model_available = False

st.sidebar.header("Model status")
with st.spinner("Checking models..."):
    # text transformer model
    if use_text:
        if os.path.isdir(text_model_dir):
            try:
                text_tokenizer, text_transformer_model = load_text_transformer_model(text_model_dir)
                text_transformer_available = True
                st.sidebar.success("Text transformer model loaded")
            except Exception as e:
                st.sidebar.warning(f"Text transformer not loaded: {e}")
        # sklearn fallback
        if not text_transformer_available and os.path.exists(text_sklearn_path):
            try:
                sklearn_pipe = load_text_sklearn_model(text_sklearn_path)
                text_sklearn_available = True
                st.sidebar.success("Text sklearn model loaded")
            except Exception as e:
                st.sidebar.warning(f"Text sklearn not loaded: {e}")
        if not text_transformer_available and not text_sklearn_available and use_text:
            st.sidebar.error("No text model found; provide transformer dir or sklearn joblib file.")
    # image model
    if enable_image:
        if os.path.exists(image_model_path):
            try:
                # attempt to load but don't pin to device permanently here
                temp_model, classes = load_image_checkpoint(image_model_path, num_classes=2)
                image_model_available = True
                image_model = temp_model  # can reuse
                st.sidebar.success("Image model loaded")
            except Exception as e:
                st.sidebar.error(f"Failed to load image model: {e}")
        else:
            st.sidebar.warning("Image model checkpoint not found at path (update path if needed).")

# ---------------------
# Text inference flow
# ---------------------
text_results_df = None
if 'text_results' not in st.session_state:
    st.session_state['text_results'] = None

if uploaded_text is not None and infer_text_button:
    # read dataset
    try:
        # attempt to read CSV/TSV intelligently
        uploaded_bytes = uploaded_text.read()
        uploaded_text.seek(0)
        # try common separators
        try:
            df = pd.read_csv(io.BytesIO(uploaded_bytes))
        except Exception:
            try:
                df = pd.read_csv(io.BytesIO(uploaded_bytes), sep='\t')
            except Exception:
                df = pd.read_json(io.BytesIO(uploaded_bytes), lines=True)
        st.write("Loaded dataset. Columns:", list(df.columns))
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        df = None

    if df is not None:
        # choose text column
        cols = list(df.columns)
        text_col = st.selectbox("Choose text column", cols, index=0)
        st.write("Preview:")
        st.dataframe(df[[text_col]].head(text_preview_n))

        texts = df[text_col].astype(str).tolist()
        st.info(f"Running inference on {len(texts)} rows (this may take some time).")
        progress = st.progress(0)
        preds = []
        t0 = time.time()
        # choose predictor
        try:
            if text_transformer_available:
                # batch predict
                batch_size_local = int(min(64, max(8, batch_size)))
                out = predict_texts_transformer(text_tokenizer, text_transformer_model, texts, batch_size=batch_size_local)
            elif text_sklearn_available:
                out = predict_texts_sklearn(sklearn_pipe, texts)
            else:
                st.error("No text model available to run inference.")
                out = []
        except Exception as e:
            st.error(f"Text inference failed: {e}")
            out = []

        # attach results
        if out:
            preds_list = [r["pred"] for r in out]
            probs_list = [r["prob"] for r in out]
            df_res = df.copy()
            df_res["_pred"] = preds_list
            df_res["_prob"] = probs_list
            st.session_state['text_results'] = df_res
            text_results_df = df_res
            t1 = time.time()
            st.success(f"Text inference complete — {len(df_res)} rows in {t1-t0:.1f}s")
        else:
            st.warning("No predictions were produced.")

# show text results if available
if st.session_state.get('text_results') is not None:
    df_res = st.session_state['text_results']
    st.subheader("Text results summary")
    total = len(df_res)
    counts = df_res["_pred"].value_counts().to_dict()
    real_count = counts.get(0, 0)
    synth_count = counts.get(1, 0)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total rows", total)
    col2.metric("Predicted real", f"{real_count} ({real_count/total*100:.2f}%)")
    col3.metric("Predicted synthetic", f"{synth_count} ({synth_count/total*100:.2f}%)")

    # pie chart
    fig1, ax1 = plt.subplots(figsize=(4,4))
    ax1.pie([real_count, synth_count], labels=["real","synthetic"], autopct='%1.1f%%', startangle=140)
    ax1.axis('equal')
    st.pyplot(fig1)

    # show low confidence / suspect examples
    suspect = df_res[df_res["_prob"] < confidence_threshold]
    st.subheader(f"Suspect / low-confidence text rows (prob < {confidence_threshold:.2f}) — {len(suspect)}")
    if not suspect.empty:
        st.dataframe(suspect[[c for c in suspect.columns if c not in ['_pred','_prob']] + ['_pred','_prob']].head(200))
    else:
        st.write("No low-confidence text examples.")

# ---------------------
# Image inference flow
# ---------------------
if 'image_results' not in st.session_state:
    st.session_state['image_results'] = None

if (uploaded_images is not None or local_image_folder) and infer_image_button:
    # gather files
    paths = []
    if uploaded_images is not None:
        try:
            fp = save_uploadedfile_to_tempfile(uploaded_images)
            # if it's a zip, extract and gather images
            if uploaded_images.name.lower().endswith(".zip"):
                paths = gather_image_paths_from_zip(fp)
            else:
                # not zip — try treat as a single archive
                try:
                    paths = gather_image_paths_from_zip(fp)
                except Exception:
                    st.error("Uploaded file not recognized as a zip of images.")
                    paths = []
        except Exception as e:
            st.error(f"Failed to save uploaded file: {e}")
            paths = []
    if local_image_folder:
        if os.path.isdir(local_image_folder):
            for root,_,fnames in os.walk(local_image_folder):
                for fn in fnames:
                    if fn.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")):
                        paths.append(os.path.join(root, fn))
        else:
            st.warning("Local image folder not found or not a directory.")

    if not paths:
        st.error("No images found to run inference.")
    else:
        st.info(f"Running image inference on {len(paths)} images. Model available: {image_model_available}")
        # ensure image model available
        if not image_model_available:
            st.error("Image model not available. Provide a valid checkpoint path in the left panel.")
        else:
            # predict
            t0 = time.time()
            try:
                out = predict_images(image_model, paths, img_size=img_size, batch_size=int(batch_size))
            except Exception as e:
                st.error(f"Image inference failed: {e}")
                out = []
            if out:
                df_img = pd.DataFrame(out)
                st.session_state['image_results'] = df_img
                t1 = time.time()
                st.success(f"Image inference complete — {len(df_img)} images in {t1-t0:.1f}s")
            else:
                st.warning("No image predictions produced.")

# show image results
if st.session_state.get('image_results') is not None:
    df_img = st.session_state['image_results']
    st.subheader("Image results summary")
    total = len(df_img)
    counts = df_img["pred"].value_counts().to_dict()
    real_count = counts.get(0, 0)
    synth_count = counts.get(1, 0)
    c1,c2,c3 = st.columns(3)
    c1.metric("Total images", total)
    c2.metric("Predicted real", f"{real_count} ({real_count/total*100:.2f}%)")
    c3.metric("Predicted synth", f"{synth_count} ({synth_count/total*100:.2f}%)")

    fig2, ax2 = plt.subplots(figsize=(4,4))
    ax2.pie([real_count, synth_count], labels=["real","synthetic"], autopct='%1.1f%%', startangle=140)
    ax2.axis('equal')
    st.pyplot(fig2)

    suspect_img = df_img[df_img["prob"] < confidence_threshold]
    st.subheader(f"Suspect / low-confidence images (prob < {confidence_threshold:.2f}) — {len(suspect_img)}")
    if not suspect_img.empty:
        st.dataframe(suspect_img.head(200))
        # show small gallery of first few suspect images
        st.write("Sample suspect images")
        cols = st.columns(6)
        for i, row in suspect_img.head(12).iterrows():
            try:
                img = Image.open(row["path"]).convert("RGB")
                col = cols[i % len(cols)]
                col.image(img.resize((120,120)), caption=f"pred={row['pred']} prob={row['prob']:.2f}", use_column_width=False)
            except Exception:
                pass
    else:
        st.write("No low-confidence images.")

# ---------------------
# Combined view & export
# ---------------------
st.markdown("---")
st.header("Combined results and export")

if st.session_state.get('text_results') is not None and st.session_state.get('image_results') is not None:
    st.success("Both text and image results available — showing combined summary.")
    # combine counts
    tr = st.session_state['text_results']
    ir = st.session_state['image_results']
    txt_total = len(tr)
    img_total = len(ir)
    txt_real = (tr["_pred"]==0).sum()
    txt_synth = (tr["_pred"]==1).sum()
    img_real = (ir["pred"]==0).sum()
    img_synth = (ir["pred"]==1).sum()
    st.write("Text dataset -> real/synth:", txt_real, txt_synth, "Image dataset -> real/synth:", img_real, img_synth)

    # export combined CSVs
    if st.button("Download combined text predictions (CSV)"):
        st.download_button("Download text predictions", data=tr.to_csv(index=False), file_name="text_predictions.csv", mime="text/csv")
    if st.button("Download image predictions (CSV)"):
        st.download_button("Download image predictions", data=ir.to_csv(index=False), file_name="image_predictions.csv", mime="text/csv")

else:
    st.info("Run text and/or image inferences to enable combined options and exports.")

st.markdown("---")
st.write("Notes:")
st.write("- If a model file/directory isn't found, update the path in the left panel.")
st.write("- For transformers text model the dir should contain a tokenizer + model files (the folder produced by `trainer.save_model()` or `AutoModelForSequenceClassification.from_pretrained` output).")
st.write("- For sklearn text model, the .pkl should be a pipeline or classifier with `predict_proba` for best results.")
st.write("- Image model loader tries to handle .pth saved as dict with `model_state_dict` key (your current checkpoint format).")

