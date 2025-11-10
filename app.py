"""Combined entrypoint for Flask and Streamlit demos.

Usage:
- Run Flask (legacy browser demo):
    python app.py --flask

- Run Streamlit (recommended):
    streamlit run app.py -- --streamlit

Implementation notes:
- The code exposes a Flask app (route `/` and `/predict`) and a Streamlit UI. Both share the same model loading and preprocessing helpers.
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

# Config
MODEL_PATH = os.environ.get('ASL_MODEL', 'models/sign_language_model.h5')
LABELS_PATH = os.environ.get('ASL_LABELS', 'models/labels.json')
IMG_SIZE = int(os.environ.get('ASL_IMG_SIZE', '224'))

# Optional runtime model downloader: set MODEL_URL env var (or Streamlit secret) to download large model at startup
MODEL_URL = os.environ.get('MODEL_URL')
if MODEL_URL and not Path(MODEL_PATH).exists():
    os.makedirs(Path(MODEL_PATH).parent, exist_ok=True)
    print('Downloading model from MODEL_URL...')
    try:
        r = requests.get(MODEL_URL, stream=True, timeout=60)
        r.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print('Model downloaded.')
    except Exception as e:
        print('Failed to download model from MODEL_URL:', e)


def load_model_and_labels() -> Tuple[Optional[object], Optional[dict]]:
    """Load Keras model and labels.json. Returns (model, labels) or (None, None) on failure."""
    try:
        import tensorflow as tf  # type: ignore
        from tensorflow.keras.models import load_model  # type: ignore
    except Exception:
        return None, None

    model = None
    labels = None
    if Path(MODEL_PATH).exists():
        model = load_model(MODEL_PATH, compile=False)
        try:
            model.make_predict_function()
        except Exception:
            pass
        try:
            model.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype='float32'))
        except Exception:
            pass

    if Path(LABELS_PATH).exists():
        with open(LABELS_PATH, 'r') as f:
            labels = {int(k): v for k, v in json.load(f).items()}

    return model, labels


def preprocess_pil_impl(img: Image.Image, img_size: int = 224) -> np.ndarray:
    img = img.convert('RGB')
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((img_size, img_size))
    arr = np.asarray(img).astype('float32')
    try:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore

        arr = preprocess_input(arr)
    except Exception:
        arr = arr / 127.5 - 1.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def preprocess_pil(img: Image.Image, img_size: int = 224) -> np.ndarray:
    return preprocess_pil_impl(img, img_size)


def preprocess_image_from_dataurl(data_url: str, img_size: int = 224) -> np.ndarray:
    header, encoded = data_url.split(',', 1)
    data = base64.b64decode(encoded)
    arr = np.frombuffer(data, np.uint8)
    import cv2  # type: ignore

    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Could not decode image')
    h, w = img.shape[:2]
    side = min(h, w)
    cx, cy = w // 2, h // 2
    x1 = cx - side // 2
    y1 = cy - side // 2
    crop = img[y1:y1+side, x1:x1+side]
    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_arr = img_resized.astype('float32')
    try:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore

        img_arr = preprocess_input(img_arr)
    except Exception:
        img_arr = img_arr / 127.5 - 1.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr


def predict_from_array(model, labels, arr: np.ndarray) -> Tuple[str, float]:
    preds = model.predict_on_batch(arr)
    idx = int(np.argmax(preds[0]))
    conf = float(np.max(preds[0]))
    label = labels.get(idx, str(idx)) if labels else str(idx)
    return label, conf


#########################
# Flask section
#########################

def create_flask_app():
    from flask import Flask, render_template, request, jsonify  # type: ignore

    flask_app = Flask(__name__, static_folder='static', template_folder='templates')

    MODEL, LABELS = load_model_and_labels()

    @flask_app.route('/')
    def index():
        model_exists = Path(MODEL_PATH).exists()
        return render_template('index.html', model_exists=model_exists)

    @flask_app.route('/predict', methods=['POST'])
    def predict():
        nonlocal MODEL, LABELS
        if MODEL is None:
            return jsonify({'error': 'Model not loaded on server'}), 500
        data = request.get_json(force=True)
        data_url = data.get('image')
        if not data_url:
            return jsonify({'error': 'No image provided'}), 400
        try:
            arr = preprocess_image_from_dataurl(data_url, img_size=IMG_SIZE)
            label, conf = predict_from_array(MODEL, LABELS, arr)
            return jsonify({'label': label, 'confidence': float(conf)})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @flask_app.route('/favicon.ico')
    def favicon():
        return ('', 204)

    return flask_app


#########################
# Streamlit section
#########################

def streamlit_main():
    import streamlit as st  # type: ignore
    try:
        from streamlit_webrtc import webrtc_streamer, VideoTransformerBase  # type: ignore
        have_webrtc = True
    except Exception:
        have_webrtc = False

    st.set_page_config(page_title='ASL Alphabet — Streamlit Demo', layout='centered')
    st.title('ASL Alphabet Recognition — Streamlit Demo')

    st.markdown('Upload an image (A–Z) or enable the optional webcam demo (requires streamlit-webrtc).')

    model, labels = load_model_and_labels()
    if model is None:
        st.warning('Model not loaded. Place a trained model at `models/sign_language_model.h5` and `models/labels.json`.')

    # UI placeholders for status and stats
    if 'latencies' not in st.session_state:
        st.session_state['latencies'] = []  # rolling list of recent latencies (s)
    status_placeholder = st.empty()
    stats_placeholder = st.empty()

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader('Upload image', type=['jpg', 'jpeg', 'png'])
        if uploaded is not None:
            image = Image.open(uploaded)
            st.image(image, caption='Uploaded image', use_column_width=True)
            if model is not None:
                # show pending indicator
                status_placeholder.info('Prediction pending...')
                import time
                t0 = time.time()
                try:
                    arr = preprocess_pil(image, img_size=IMG_SIZE)
                    label, conf = predict_from_array(model, labels, arr)
                finally:
                    latency = time.time() - t0
                    # maintain rolling window
                    latencies = st.session_state['latencies']
                    latencies.append(latency)
                    if len(latencies) > 20:
                        latencies.pop(0)
                    st.session_state['latencies'] = latencies
                    status_placeholder.empty()
                st.success(f'Prediction: {label} ({conf*100:.1f}%)')
                # update stats
                avg_lat = sum(st.session_state['latencies']) / len(st.session_state['latencies'])
                fps = 1.0 / avg_lat if avg_lat > 0 else 0.0
                stats_placeholder.markdown(f"**Avg latency:** {avg_lat*1000:.0f} ms — **FPS:** {fps:.1f}")

    with col2:
        st.write('Tip: center the hand, use plain background and good lighting for best results.')

    st.markdown('---')
    st.subheader('Optional: Webcam (experimental)')
    if have_webrtc:
        class ASLTransformer(VideoTransformerBase):
            def __init__(self):
                self.model, self.labels = load_model_and_labels()

            def transform(self, frame):
                img = frame.to_image()
                if self.model is None:
                    return frame
                try:
                    arr = preprocess_pil(img, img_size=IMG_SIZE)
                    label, conf = predict_from_array(self.model, self.labels, arr)
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(img)
                    draw.text((10, 10), f'{label} {conf*100:.1f}%', fill=(255, 0, 0))
                except Exception:
                    pass
                return np.array(img)

        st.info('Webcam demo enabled (streamlit-webrtc). Click Start to open webcam.')
        webrtc_streamer(key='asl-demo', video_transformer_factory=ASLTransformer)
    else:
        st.info('Webcam demo not available. Install `streamlit-webrtc` to enable it.')


def preprocess_pil(img: Image.Image, img_size: int = 224) -> np.ndarray:
    # small wrapper; reuse same preprocessing as streamlit_app
    return preprocess_pil_impl(img, img_size)


def preprocess_pil_impl(img: Image.Image, img_size: int = 224) -> np.ndarray:
    img = img.convert('RGB')
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((img_size, img_size))
    arr = np.asarray(img).astype('float32')
    try:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore

        arr = preprocess_input(arr)
    except Exception:
        arr = arr / 127.5 - 1.0
    arr = np.expand_dims(arr, axis=0)
    return arr


if __name__ == '__main__':
    # simple CLI: --flask to run Flask, --streamlit to run Streamlit via python (rare). Default: Flask
    args = sys.argv[1:]
    if '--streamlit' in args:
        # run streamlit UI inside this process (not common) -- attempt to import and run
        try:
            streamlit_main()
        except Exception as e:
            print('Failed to run Streamlit UI:', e)
    else:
        # default: run Flask app
        flask_app = create_flask_app()
        flask_app.run(host='0.0.0.0', port=5000, debug=True)
