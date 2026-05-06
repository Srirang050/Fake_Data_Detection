# ui/app_streamlit.py
"""
Streamlit UI for DataGuard - Unified Text & Image Fake/Synthetic Detector
Assumes the following files exist in ../models (project root -> models/):
 - class_map.json
 - iso_images_improved.pkl
 - iso_scaler_improved.pkl
 - iso_hybrid_logreg.pkl
 - iso_threshold.json
 - text_fake_detector_clf.pkl
 - text_fake_detector_scaler.pkl

Run from project root:
  streamlit run ui/app_streamlit.py
"""
import streamlit as st
import os, json, joblib, numpy as np, io, time
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "models")

st.set_page_config(page_title="DataGuard — Real vs Synthetic Detector", layout="wide")

st.title("DataGuard — Real vs Synthetic Detector")
st.markdown("Upload text or images to detect whether they are *real* or *AI-generated* (synthetic).")

# ------------------ helper loaders ------------------
@st.cache_resource
def load_image_models():
    out = {}
    # load class map
    cmap_path = os.path.join(MODEL_DIR, "class_map.json")
    out["class_map"] = json.load(open(cmap_path)) if os.path.exists(cmap_path) else {"real":0,"synth":1}
    # isolation forest + scaler
    iso_p = os.path.join(MODEL_DIR, "iso_images_improved.pkl")
    scaler_p = os.path.join(MODEL_DIR, "iso_scaler_improved.pkl")
    hybrid_p = os.path.join(MODEL_DIR, "iso_hybrid_logreg.pkl")
    thr_p = os.path.join(MODEL_DIR, "iso_threshold.json")
    out["iso"] = joblib.load(iso_p) if os.path.exists(iso_p) else None
    out["scaler"] = joblib.load(scaler_p) if os.path.exists(scaler_p) else None
    out["hybrid"] = joblib.load(hybrid_p) if os.path.exists(hybrid_p) else None
    out["threshold"] = (json.load(open(thr_p))["best_threshold"] if os.path.exists(thr_p) else None)
    return out

@st.cache_resource
def load_text_model():
    clf_p = os.path.join(MODEL_DIR, "text_fake_detector_clf.pkl")
    scaler_p = os.path.join(MODEL_DIR, "text_fake_detector_scaler.pkl")
    clf = joblib.load(clf_p) if os.path.exists(clf_p) else None
    scaler = joblib.load(scaler_p) if os.path.exists(scaler_p) else None
    return clf, scaler

# lazy load only when used
img_models = None
txt_model = None

tabs = st.tabs(["Text", "Image", "Audio (placeholder)"])

# ------------------ TEXT TAB ------------------
with tabs[0]:
    st.header("Text Detector")
    st.write("Paste text (e.g., a review) and click Analyze.")
    text_input = st.text_area("Text input", height=220, placeholder="Paste or type text here...")
    col1, col2 = st.columns([1,3])
    with col1:
        if st.button("Analyze Text"):
            if not text_input.strip():
                st.warning("Enter some text first.")
            else:
                # load text model
                clf, scaler = load_text_model()
                if clf is None or scaler is None:
                    st.error("Text model or scaler not found in models/ (text_fake_detector_clf.pkl, text_fake_detector_scaler.pkl)")
                else:
                    def extract_features(text):
                        words = text.split()
                        num_words = len(words)
                        unique_words = len(set(words))
                        avg_sentence_length = (sum(len(s.split()) for s in text.split('.') if s) / max(1, len([s for s in text.split('.') if s])))
                        repetition_score = num_words / (unique_words + 1)
                        return np.array([num_words, unique_words, avg_sentence_length, repetition_score]).reshape(1,-1)
                    X = extract_features(text_input)
                    Xs = scaler.transform(X)
                    prob = float(clf.predict_proba(Xs)[0,1])
                    pred = int(clf.predict(Xs)[0])
                    st.success(f"Prediction: **{'SYNTHETIC' if pred==1 else 'REAL'}**")
                    st.write(f"Probability (synthetic): **{prob:.4f}**")
                    st.markdown("**Extracted features**")
                    st.json({
                        "num_words": int(X[0,0]),
                        "unique_words": int(X[0,1]),
                        "avg_sentence_length": float(round(X[0,2],3)),
                        "repetition_score": float(round(X[0,3],3))
                    })
    with col2:
        st.info("Notes: The text detector was trained on IMDB reviews and GPT-2 generated synthetic samples. It uses simple textual features and a logistic regression classifier.")

# ------------------ IMAGE TAB ------------------
with tabs[1]:
    st.header("Image Detector")
    st.write("Upload an image (jpg, png). The app will extract ResNet features and run the IsolationForest + hybrid model.")
    uploaded = st.file_uploader("Upload image file", type=["jpg","jpeg","png"])
    st.write("Tip: Use test images from `data/images/test/real` and `data/images/test/synth` for demo.")
    if uploaded is not None:
        try:
            img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        except Exception as e:
            st.error("Failed to read image: " + str(e))
            img = None
        if img:
            st.image(img, width=300)
            if st.button("Analyze Image"):
                st.info("Processing image — this may take a few seconds (loads ResNet).")
                # load models
                img_models = load_image_models()
                iso = img_models["iso"]; scaler = img_models["scaler"]; hybrid = img_models["hybrid"]
                threshold = img_models["threshold"]
                if iso is None or scaler is None:
                    st.error("Image models not found in models/. Required: iso_images_improved.pkl and iso_scaler_improved.pkl")
                else:
                    # lazy import torchvision/resnet
                    try:
                        import torch
                        from torchvision import transforms
                        from torchvision.models import resnet50
                    except Exception as e:
                        st.error("torch/torchvision not found or failed to import. Install them in your venv. Error: " + str(e))
                        raise
                    preprocess = transforms.Compose([
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                    t = preprocess(img).unsqueeze(0)  # (1,C,H,W)
                    # build resnet backbone
                    res = resnet50(pretrained=True)
                    res.fc = torch.nn.Identity()
                    res.eval()
                    with torch.no_grad():
                        feat = res(t).cpu().numpy()  # (1,2048)
                    feat_s = scaler.transform(feat)
                    iso_score = float(-iso.score_samples(feat_s)[0])
                    st.write(f"Iso anomaly score (higher => more anomalous): **{iso_score:.6f}**")
                    # hybrid handling
                    hybrid_prob = None
                    try:
                        # hybrid could be a tuple (lr, scaler2) or a single model depending on save
                        if isinstance(hybrid, tuple) or isinstance(hybrid, list):
                            lr, scaler2 = hybrid
                            hybrid_prob = float(lr.predict_proba(scaler2.transform([[iso_score]]))[:,1][0])
                        else:
                            # if hybrid is stored as a single estimator expecting 1-D feature, try naive
                            hybrid_prob = None
                    except Exception:
                        hybrid_prob = None
                    if hybrid_prob is not None:
                        st.success(f"Hybrid probability (synthetic): **{hybrid_prob:.4f}**")
                        st.write("Final prediction:", "**SYNTHETIC**" if hybrid_prob>0.5 else "**REAL**")
                    else:
                        st.write("Hybrid model not available or not in expected format. Using threshold.")
                        if threshold is not None:
                            pred = 1 if iso_score >= threshold else 0
                            st.success(f"Prediction (threshold): **{'SYNTHETIC' if pred==1 else 'REAL'}** (threshold={threshold:.6f})")
                        else:
                            st.warning("No threshold stored. You can inspect iso_score and decide.")
                    # optional: download JSON
                    result = {
                        "timestamp": time.time(),
                        "iso_score": iso_score,
                        "hybrid_prob": hybrid_prob,
                    }
                    st.download_button("Download result (JSON)", data=json.dumps(result, indent=2), file_name="dg_result.json")
    else:
        st.info("No image uploaded — upload a sample to run detection.")

# ------------------ AUDIO TAB (placeholder) ------------------
with tabs[2]:
    st.header("Audio Detector (placeholder)")
    st.info("Audio module will be added later. Upload audio to test once model is integrated.")
    audio = st.file_uploader("Upload audio (.wav/.mp3)", type=["wav","mp3"])
    if audio:
        st.audio(audio)
        st.write("Audio detection not implemented yet.")

# footer
st.markdown("---")
st.write("Models loaded from:", MODEL_DIR)
st.write("If models are missing or the app fails to import torch, activate your `dataguard` venv and install dependencies.")
