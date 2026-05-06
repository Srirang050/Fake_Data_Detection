#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="dataguard"     # <- change if you used a different conda env
DATA_CSV="data/combined_texts.csv"
IMG_DIR="data/images"
TEXT_MODEL_DIR="models/text_fake_detector_roberta"
IMAGE_MODEL_DIR="models/image_fake_detector_try1"
RESULTS_DIR="results"

# activate env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

mkdir -p "$RESULTS_DIR"

echo "1) Convert safetensors -> pytorch (if you have model.safetensors)"
python convert_safetensors_v2.py --model-dir "$TEXT_MODEL_DIR" || echo "convert step failed (maybe already converted)"

echo "2) Train text (small quick run) — adjust args if you want full training"
PYTHONPATH=. python scripts/train_text_detector.py \
  --input "$DATA_CSV" \
  --text-col text --label-col label \
  --model_name roberta-base \
  --output_dir "$TEXT_MODEL_DIR" \
  --num_train_epochs 1 \
  --batch_size 8

echo "3) Train image (fast small run)"
python scripts/train_image_detector_improved.py \
  --data "$IMG_DIR" \
  --out "$IMAGE_MODEL_DIR" \
  --epochs 1 \
  --batch-size 8 \
  --img-size 128 \
  --num-workers 0 \
  --freeze-backbone

echo "4) Run batch inference on text (example)"
PYTHONPATH=. python scripts/infer_text_batch.py \
  --model "$TEXT_MODEL_DIR" \
  --input "$DATA_CSV" \
  --out "$RESULTS_DIR/preds_text.csv" \
  --text-col text \
  --batch-size 64

echo "5) Run Streamlit UI (open browser at http://localhost:8501)"
streamlit run app_text_detector.py --server.port 8501