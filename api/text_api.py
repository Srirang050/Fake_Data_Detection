#!/usr/bin/env python3
# api/text_api.py
"""
Combined Text + Image API for DataGuard.
Place this file at api/text_api.py (project layout assumed).
Serves UI from ../ui and models from ../models.
"""

import os
import io
import json
import traceback
import joblib
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory

# optional ML imports
try:
    import torch
    import torchvision.models as tv_models
    import torchvision.transforms as T
except Exception:
    torch = None
    tv_models = None
    T = None

# -------------------------
# Paths / config
# -------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
UI_DIR = os.path.join(PROJECT_ROOT, "ui")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

TEXT_CLF_PKL = os.path.join(MODEL_DIR, "text_fake_detector_clf.pkl")
TEXT_SCALER_PKL = os.path.join(MODEL_DIR, "text_fake_detector_scaler.pkl")

ISO_PKL = os.path.join(MODEL_DIR, "iso_images_improved.pkl")
ISO_SCALER_PKL = os.path.join(MODEL_DIR, "iso_scaler_improved.pkl")
ISO_THRESHOLD_JSON = os.path.join(MODEL_DIR, "iso_threshold.json")
HYBRID_PKL = os.path.join(MODEL_DIR, "iso_hybrid_logreg.pkl")  # may be tuple (LR, scaler)

# -------------------------
# Helpers
# -------------------------
def safe_joblib_load(path):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        print(f"❌ joblib.load failed for {path} — traceback below")
        traceback.print_exc()
        return None

# -------------------------
# Load text classifier + scaler
# -------------------------
text_clf = safe_joblib_load(TEXT_CLF_PKL)
text_scaler = safe_joblib_load(TEXT_SCALER_PKL)
print("Loaded text classifier:", TEXT_CLF_PKL if text_clf is not None else "MISSING")
print("Loaded text scaler:", TEXT_SCALER_PKL if text_scaler is not None else "MISSING")

# -------------------------
# Build ResNet embedding model (robust across torchvision versions)
# -------------------------
def build_resnet_embedding_model(device="cpu"):
    """
    Returns (resnet_model, transform) or (None, None) on failure.
    The model's fc is replaced with Identity so output is embedding vector.
    """
    if torch is None:
        print("torch not available; cannot build resnet.")
        return None, None

    try:
        # Try to use weights API (torchvision >= 0.13)
        Weights = getattr(tv_models, "ResNet50_Weights", None)
        if Weights is not None:
            # prefer DEFAULT if exists else IMAGENET1K_V2 or IMAGENET1K_V1
            weight_enum = getattr(Weights, "DEFAULT", None) or getattr(Weights, "IMAGENET1K_V2", None) or getattr(Weights, "IMAGENET1K_V1", None)
            if weight_enum is not None:
                resnet = tv_models.resnet50(weights=weight_enum)
                try:
                    transform = weight_enum.transforms()
                except Exception:
                    # fallback transform if new API not present
                    transform = T.Compose([
                        T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                    ])
            else:
                resnet = tv_models.resnet50(pretrained=True)
                transform = T.Compose([
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
        else:
            # older torchvision or missing weights enum
            resnet = tv_models.resnet50(pretrained=True)
            transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

        # replace final layer with identity to get embeddings
        resnet.fc = torch.nn.Identity()
        resnet.eval()
        resnet.to(device)
        return resnet, transform
    except Exception:
        print("Failed to build ResNet via torchvision. Traceback:")
        traceback.print_exc()
        return None, None

# attempt to build resnet
resnet_model, resnet_transform = build_resnet_embedding_model(device="cpu")
if resnet_model is not None:
    print("✅ ResNet50 embedding model loaded (cpu).")
else:
    print("❌ ResNet50 embedding model not available.")

# -------------------------
# Load image isolation models
# -------------------------
iso_model = safe_joblib_load(ISO_PKL)
iso_scaler = safe_joblib_load(ISO_SCALER_PKL)

iso_threshold = None
if os.path.exists(ISO_THRESHOLD_JSON):
    try:
        iso_threshold = json.load(open(ISO_THRESHOLD_JSON, "r"))
    except Exception:
        print("Failed to read iso threshold JSON; ignoring.")
        traceback.print_exc()
        iso_threshold = None

print("Iso model:", ISO_PKL if iso_model is not None else "MISSING")
print("Iso scaler:", ISO_SCALER_PKL if iso_scaler is not None else "MISSING")
print("Iso threshold:", iso_threshold)

# -------------------------
# Load hybrid logistic/reg (could be tuple or a single object)
# -------------------------
raw_hybrid = safe_joblib_load(HYBRID_PKL)
hybrid_lr = None
hybrid_scaler = None
if raw_hybrid is None:
    print("Hybrid LR: MISSING")
else:
    # If tuple/list: try to detect classifier and scaler elements
    if isinstance(raw_hybrid, (tuple, list)):
        for el in raw_hybrid:
            if hasattr(el, "predict") and hybrid_lr is None:
                hybrid_lr = el
            elif hasattr(el, "transform") and hybrid_scaler is None:
                hybrid_scaler = el
    elif hasattr(raw_hybrid, "predict"):
        hybrid_lr = raw_hybrid
    elif isinstance(raw_hybrid, dict):
        for k, v in raw_hybrid.items():
            if hasattr(v, "predict") and hybrid_lr is None:
                hybrid_lr = v
            if hasattr(v, "transform") and hybrid_scaler is None:
                hybrid_scaler = v

print("Hybrid LR loaded:", bool(hybrid_lr))
print("Hybrid scaler loaded:", bool(hybrid_scaler))

# -------------------------
# Flask app + routes
# -------------------------
app = Flask(__name__, static_folder=None)

# serve UI index at root
@app.route("/", methods=["GET"])
def serve_ui_index():
    if os.path.exists(os.path.join(UI_DIR, "index.html")):
        return send_from_directory(UI_DIR, "index.html")
    return "<html><body><h3>UI not found — place ui/index.html</h3></body></html>", 404

# serve ui assets (example: /ui/whatever.css)
@app.route("/ui/<path:filename>")
def serve_ui_file(filename):
    return send_from_directory(UI_DIR, filename)

# health check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "text_model_loaded": text_clf is not None and text_scaler is not None,
        "resnet_loaded": resnet_model is not None,
        "iso_model_loaded": iso_model is not None,
        "hybrid_loaded": hybrid_lr is not None
    })

# -------------------------
# Text endpoint
# -------------------------
def extract_simple_features(text: str):
    words = text.split()
    num_words = len(words)
    unique_words = len(set(words))
    sents = [s.strip() for s in text.split('.') if s.strip()]
    avg_sentence_length = float(np.mean([len(s.split()) for s in sents])) if sents else 0.0
    repetition_score = num_words / (unique_words + 1)
    return [num_words, unique_words, avg_sentence_length, repetition_score]

# try to import local perplexity helper (if project provides it)
try:
    from api.text_perplexity import perplexity
except Exception:
    perplexity = None

@app.route("/predict_text", methods=["POST"])
def predict_text():
    data = request.get_json(force=True, silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Please POST JSON with key 'text'."}), 400

    text = str(data["text"])
    feats_simple = extract_simple_features(text)

    if perplexity is None:
        return jsonify({"error": "perplexity helper not available on server."}), 500

    try:
        ppls = perplexity([text], batch_size=1)
        ppl = float(ppls[0])
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Failed to compute perplexity", "detail": str(e)}), 500

    if text_scaler is None or text_clf is None:
        return jsonify({"error": "text model or scaler missing on server"}), 500

    try:
        feats = np.array([feats_simple + [ppl]], dtype=float)
        feats_scaled = text_scaler.transform(feats)
        pred = int(text_clf.predict(feats_scaled)[0])
        proba_fake = float(text_clf.predict_proba(feats_scaled)[:, 1][0]) if hasattr(text_clf, "predict_proba") else None
        label = "fake" if pred == 1 else "real"
        return jsonify({
            "label": label,
            "probability_fake": proba_fake,
            "perplexity": float(ppl),
            "features": {
                "num_words": int(feats_simple[0]),
                "unique_words": int(feats_simple[1]),
                "avg_sentence_length": float(feats_simple[2]),
                "repetition_score": float(feats_simple[3])
            }
        })
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "failed during text predict", "detail": "see server logs"}), 500

# -------------------------
# Image endpoint
# -------------------------
@app.route("/predict_image", methods=["POST"])
def predict_image():
    # expect multipart/form-data with 'file'
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded (field name 'file')"}), 400
    f = request.files["file"]
    try:
        img = Image.open(io.BytesIO(f.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": "Failed to read image", "detail": str(e)}), 400

    # compute embedding
    if resnet_model is None or resnet_transform is None:
        return jsonify({"error": "ResNet embedding model not available on server."}), 500

    try:
        # transform expects PIL.Image
        img_t = resnet_transform(img).unsqueeze(0)  # 1 x C x H x W
        with torch.no_grad():
            emb_t = resnet_model(img_t)  # 1 x D (D likely 2048)
        emb = emb_t.cpu().numpy().astype(np.float32)
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Failed to compute embedding", "detail": "see server logs"}), 500

    # iso pipeline: need iso_model to score embeddings
    if iso_model is None:
        # if iso missing but hybrid exists, we still need an iso_score; so we require iso.
        return jsonify({"error": "ISO model or scaler missing"}), 500

    try:
        if hasattr(iso_model, "decision_function"):
            iso_score = float(iso_model.decision_function(emb)[0])
        elif hasattr(iso_model, "score_samples"):
            iso_score = float(iso_model.score_samples(emb)[0])
        else:
            return jsonify({"error": "iso model has no scoring method"}), 500
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "iso model failed to score embedding"}), 500

    # try to scale iso score
    iso_scaled = None
    if iso_scaler is not None:
        try:
            iso_scaled = iso_scaler.transform(np.array([[iso_score]], dtype=float))
        except Exception:
            try:
                iso_scaled = iso_scaler.transform(np.array([iso_score], dtype=float).reshape(1, -1))
            except Exception:
                iso_scaled = None

    # final decision (prefer hybrid_lr if available)
    final_label = None
    final_prob = None
    try:
        if hybrid_lr is not None:
            if hybrid_scaler is not None:
                X_for_lr = hybrid_scaler.transform(np.array([[iso_score]], dtype=float))
            elif iso_scaled is not None:
                X_for_lr = iso_scaled
            else:
                X_for_lr = np.array([[iso_score]], dtype=float)

            pred = int(hybrid_lr.predict(X_for_lr)[0])
            final_label = "fake" if pred == 1 else "real"
            if hasattr(hybrid_lr, "predict_proba"):
                final_prob = float(hybrid_lr.predict_proba(X_for_lr)[:, 1][0])
        else:
            # use iso_threshold if present; assume iso_score < threshold -> fake
            if iso_threshold is not None:
                try:
                    if isinstance(iso_threshold, dict):
                        tval = float(iso_threshold.get("threshold", iso_threshold.get("t", 0.0)))
                    else:
                        tval = float(iso_threshold)
                except Exception:
                    tval = float(iso_threshold)
                final_label = "fake" if iso_score < tval else "real"
            else:
                final_label = "fake" if iso_score < 0 else "real"
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Failed to compute final label", "detail": traceback.format_exc()}), 500

    return jsonify({
        "label": final_label,
        "probability_fake": final_prob,
        "iso_score": float(iso_score),
        "features": {
            "embedding_shape": list(emb.shape)
        }
    })

# -------------------------
# run server
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    print("Starting DataGuard UI+API on http://0.0.0.0:%d ..." % port)
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("UI_DIR:", UI_DIR)
    print("MODEL_DIR:", MODEL_DIR)
    app.run(host="0.0.0.0", port=port)
