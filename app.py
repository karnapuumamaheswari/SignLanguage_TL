"""
Unified entrypoint for ASL Alphabet Recognition
- Streamlit UI (default on Streamlit Cloud)
- Flask API (optional for local browser testing)

Usage locally:
  streamlit run app.py              # Recommended
  python app.py --flask             # Optional Flask mode
"""

from __future__ import annotations
import os
import sys
import json
import base64
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import requests

# --- Configuration ---
MODEL_PATH = os.environ.get("ASL_MODEL", "models/sign_language_model.h5")
LABELS_PATH = os.environ.get("ASL_LABELS", "models/labels.json")
IMG_SIZE = int(os.environ.get("ASL_IMG_SIZE", "224"))
MODEL_URL = os.environ.get("MODEL_URL")  # Optional model download URL

# --- Optional: Download model if URL provided ---
if MODEL_URL and not Path(MODEL_PATH).exists():
    os.makedirs(Path(MODEL_PATH).parent, exist_ok=True)
    print("Downloading model from MODEL_URL...")
    try:
        r = requests.get(MODEL_URL, stream=True, timeout=60)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("✅ Model downloaded successfully.")
    except Exception as e:
        print("❌ Failed to download model:", e)


# --- Helper: Load TensorFlow model & labels ---
def load_model_and_labels() -> Tuple[Optional[object], Optional[dict]]:
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
    except Exception:
        print("TensorFlow not found — please ensure it's installed.")
        return None, None

    model, labels = None, None

    if Path(MODEL_PATH).exists():
        model = load_model(MODEL_PATH, compile=False)
        try:
            model.make_predict_function()
            model.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32"))
        except Exception:
            pass

    if Path(LABELS_PATH).exists():
        with open(LABELS_PATH, "r") as f:
            labels = {int(k): v for k, v in json.load(f).items()}

    return model, labels


# --- Image Preprocessing ---
def preprocess_pil_impl(img: Image.Image, img_size: int = 224) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((img_size, img_size))
    arr = np.asarray(img).astype("float32")
    try:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        arr = preprocess_input(arr)
    except Exception:
        arr = arr / 127.5 - 1.0
    return np.expand_dims(arr, axis=0)


def preprocess_image_from_dataurl(data_url: str, img_size: int = 224) -> np.ndarray:
    header, encoded = data_url.split(",", 1)
    data = base64.b64decode(encoded)
    arr = np.frombuffer(data, np.uint8)
    import cv2

    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    h, w = img.shape[:2]
    side = min(h, w)
    cx, cy = w // 2, h // 2
    crop = img[cy - side // 2 : cy + side // 2, cx - side // 2 : cx + side // 2]
    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_arr = img_resized.astype("float32")
    try:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        img_arr = preprocess_input(img_arr)
    except Exception:
        img_arr = img_arr / 127.5 - 1.0
    return np.expand_dims(img_arr, axis=0)


def predict_from_array(model, labels, arr: np.ndarray) -> Tuple[str, float]:
    preds = model.predict_on_batch(arr)
    idx = int(np.argmax(preds[0]))
    conf = float(np.max(preds[0]))
    label = labels.get(idx, str(idx)) if labels else str(idx)
    return label, conf


# -------------------------------------------------------------
# 🧩 Flask app (optional local demo)
# -------------------------------------------------------------
def create_flask_app():
    from flask import Flask, render_template, request, jsonify

    flask_app = Flask(__name__, static_folder="static", template_folder="templates")
    MODEL, LABELS = load_model_and_labels()

    @flask_app.route("/")
    def index():
        model_exists = Path(MODEL_PATH).exists()
        return render_template("index.html", model_exists=model_exists)

    @flask_app.route("/predict", methods=["POST"])
    def predict():
        nonlocal MODEL, LABELS
        if MODEL is None:
            return jsonify({"error": "Model not loaded on server"}), 500
        data = request.get_json(force=True)
        data_url = data.get("image")
        if not data_url:
            return jsonify({"error": "No image provided"}), 400
        try:
            arr = preprocess_image_from_dataurl(data_url, img_size=IMG_SIZE)
            label, conf = predict_from_array(MODEL, LABELS, arr)
            return jsonify({"label": label, "confidence": float(conf)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return flask_app


# -------------------------------------------------------------
# 🧠 Streamlit app (cloud & desktop)
# -------------------------------------------------------------
def streamlit_main():
    import streamlit as st
    try:
        from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
        have_webrtc = True
    except Exception:
        have_webrtc = False

    st.set_page_config(page_title="ASL Alphabet Recognition", layout="centered")
    st.title("🤟 ASL Alphabet Recognition — Streamlit Demo")
    st.markdown("Upload an image (A–Z) or use the webcam demo below (if enabled).")

    model, labels = load_model_and_labels()
    if model is None:
        st.error("Model not loaded! Please ensure your `.h5` and `labels.json` files exist in `/models/`.")
        st.stop()

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        import time
        t0 = time.time()
        arr = preprocess_pil_impl(image, img_size=IMG_SIZE)
        label, conf = predict_from_array(model, labels, arr)
        latency = time.time() - t0
        st.success(f"**Prediction:** {label} ({conf*100:.1f}%)")
        st.caption(f"⏱ Avg inference time: {latency*1000:.1f} ms")

    st.divider()
    st.subheader("Optional: Webcam demo (requires `streamlit-webrtc`)")

    if have_webrtc:
        class ASLTransformer(VideoTransformerBase):
            def __init__(self):
                self.model, self.labels = load_model_and_labels()

            def transform(self, frame):
                img = frame.to_image()
                try:
                    arr = preprocess_pil_impl(img, img_size=IMG_SIZE)
                    label, conf = predict_from_array(self.model, self.labels, arr)
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(img)
                    draw.text((10, 10), f"{label} {conf*100:.1f}%", fill=(255, 0, 0))
                except Exception:
                    pass
                return np.array(img)

        st.info("🎥 Click Start below to use webcam for live prediction.")
        webrtc_streamer(key="asl-demo", video_transformer_factory=ASLTransformer)
    else:
        st.info("Install `streamlit-webrtc` to enable webcam demo.")


# -------------------------------------------------------------
# 🚀 Entry Point (auto Streamlit for Cloud)
# -------------------------------------------------------------
if __name__ == "__main__":
    running_in_streamlit = os.environ.get("STREAMLIT_SERVER_RUN_ONCE", None) is not None

    if running_in_streamlit:
        # ✅ Force Streamlit UI when deployed on Streamlit Cloud
        streamlit_main()
    else:
        # 🧩 Local usage: Flask or Streamlit manually
        args = sys.argv[1:]
        if "--flask" in args:
            flask_app = create_flask_app()
            flask_app.run(host="0.0.0.0", port=5000, debug=True)
        elif "--streamlit" in args:
            streamlit_main()
        else:
            print("Usage:")
            print("  python app.py --flask       # Run Flask locally")
            print("  streamlit run app.py        # Run Streamlit (recommended)")
