"""Flask web app to run ASL alphabet predictions from the browser webcam.

Endpoints:
- GET /        -> serves the web UI
- POST /predict -> accepts JSON {image: dataURL} and returns JSON {label, confidence}

Run:
    flask run --host=0.0.0.0 --port=5000
or:
    python app.py

Note: make sure your Python environment has TensorFlow and OpenCV installed.
"""
import os
import io
import json
import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, static_folder='static', template_folder='templates')

# Config: paths to model and labels
MODEL_PATH = os.environ.get('ASL_MODEL', 'models/sign_language_model.h5')
LABELS_PATH = os.environ.get('ASL_LABELS', 'models/labels.json')
IMG_SIZE = int(os.environ.get('ASL_IMG_SIZE', '224'))

MODEL = None
LABELS = None

def load_model_and_labels():
    global MODEL, LABELS
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    except Exception:
        MODEL = None
        LABELS = None
        return

    if os.path.exists(MODEL_PATH):
        # load without compiling to be faster for inference-only use
        MODEL = load_model(MODEL_PATH, compile=False)
    else:
        MODEL = None

    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, 'r') as f:
            LABELS = {int(k): v for k, v in json.load(f).items()}
    else:
        LABELS = None

    # Warm up the model to avoid the first-request latency spike and
    # prepare the predict function (helps some TF/Keras versions).
    if MODEL is not None:
        try:
            # make the predict function (for some TF versions)
            MODEL.make_predict_function()
        except Exception:
            pass
        try:
            import numpy as _np
            dummy = _np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype='float32')
            # run a single dummy prediction to warm up kernels
            MODEL.predict(dummy)
        except Exception:
            pass


def preprocess_image_from_dataurl(data_url, img_size=224):
    # data_url = 'data:image/jpeg;base64,/9j/4AAQ...'
    header, encoded = data_url.split(',', 1)
    data = base64.b64decode(encoded)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Could not decode image')
    # center square crop
    h, w = img.shape[:2]
    side = min(h, w)
    cx, cy = w // 2, h // 2
    x1 = cx - side // 2
    y1 = cy - side // 2
    crop = img[y1:y1+side, x1:x1+side]
    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_arr = img_resized.astype('float32')
    # use MobileNetV2 preprocessing if available
    try:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        img_arr = preprocess_input(img_arr)
    except Exception:
        img_arr = img_arr / 127.5 - 1.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr


@app.route('/')
def index():
    model_exists = os.path.exists(MODEL_PATH)
    return render_template('index.html', model_exists=model_exists)


@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({'error': 'Model not loaded on server'}), 500

    data = request.get_json(force=True)
    data_url = data.get('image')
    if not data_url:
        return jsonify({'error': 'No image provided'}), 400

    try:
        arr = preprocess_image_from_dataurl(data_url, img_size=IMG_SIZE)
        # use predict_on_batch for slightly lower overhead on single-batch inference
        preds = MODEL.predict_on_batch(arr)
        idx = int(np.argmax(preds[0]))
        conf = float(np.max(preds[0]))
        label = LABELS.get(idx, str(idx)) if LABELS else str(idx)
        return jsonify({'label': label, 'confidence': float(conf)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/favicon.ico')
def favicon():
    # return empty response for favicon requests to avoid 404 log spam
    return ('', 204)


if __name__ == '__main__':
    load_model_and_labels()
    # allow running with `python app.py`
    app.run(host='0.0.0.0', port=5000, debug=True)
