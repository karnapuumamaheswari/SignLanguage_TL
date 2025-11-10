<<<<<<< HEAD
# Sign Language Alphabet Recognition using Transfer Learning

This project demonstrates how to use Transfer Learning (MobileNetV2) to classify American Sign Language (ASL) alphabets (A-Z) and run real-time predictions using a webcam.

## Contents
<!-- - `train.py` - Training script that builds and trains a MobileNetV2-based classifier.
- `prepare_data.py` - Helper to split the Kaggle ASL dataset into `train` / `val` folders.
- `realtime.py` - Real-time webcam demo using the trained model. -->
## SignLanguage_TL — ASL Alphabet Recognition

Inspiring people and machines to communicate.

This repository demonstrates a compact ASL alphabet recognizer using MobileNetV2 transfer learning. It includes training utilities, evaluation scripts, a desktop OpenCV demo, a Flask-based web demo (legacy), and a new Streamlit app for easy deployment.

Repository layout
- `train.py` — Training script (MobileNetV2 base + custom head).
- `prepare_data.py` — Split raw dataset into `data/train` and `data/val`.
- `evaluate.py` — Run evaluation on the validation set.
- `predict_webcam.py` / `realtime.py` — OpenCV desktop realtime demo.
- `app.py` — Flask web server (legacy browser demo; you can keep or remove it).
- `streamlit_app.py` — New Streamlit demo (image upload + optional webcam via `streamlit-webrtc`).
- `templates/`, `static/` — original Flask frontend assets (can remain if you want both demos).
- `models/` — store trained model(s) and `labels.json` (recommended: do not commit large models to git).

Quick start (recommended)
1. Use Python 3.10 or 3.11 (TensorFlow wheels are best supported there).
 # Sign Language Alphabet Recognition (Transfer Learning)

This repository implements an ASL (American Sign Language) alphabet recognizer using transfer learning (MobileNetV2). It includes scripts for training and evaluation, a desktop OpenCV demo, and two interfaces for quick interactive demos:

- A combined Flask + Streamlit codebase (`app.py`) offering both a legacy Flask web demo and a Streamlit demo.
- A modern, interactive browser demo (Flask frontend) with a tabbed Camera / Upload UI (drag-and-drop, preview, spinner, prediction history).

Why this repo
- Small, practical example of transfer learning (MobileNetV2 base + small classification head).
- Multiple ways to demo: desktop webcam, browser camera, or image upload.
- Simple deployment options: run locally, push to GitHub + Streamlit Cloud, or host the model separately and let the app download it at runtime.

Repository structure (important files)
- `train.py` — training script using MobileNetV2 as a base.
- `prepare_data.py` — helper to split your dataset into `data/train` and `data/val`.
- `evaluate.py` — produce classification reports and confusion matrices.
- `predict_webcam.py` / `realtime.py` — OpenCV desktop realtime demo.
- `app.py` — Combined entrypoint: exposes a Flask server (`/` and `/predict`) and Streamlit UI (callable via Streamlit). This is the primary entrypoint now.
- `templates/`, `static/` — Flask frontend assets (the interactive tabbed UI lives here).
- `models/` — models and `labels.json`. Keep this directory out of git for large models; use `MODEL_URL` to download at runtime instead.
- `.gitignore` — updated to ignore `models/`, virtualenvs, caches and model files (`*.h5`).

Requirements
- This project uses TensorFlow/Keras, OpenCV, Flask, Streamlit (optional), and supporting packages. The `requirements.txt` in the repo includes the main deps (additionally `requests` is used by the runtime downloader).
- TensorFlow compatibility: use Python 3.10 or 3.11 for easiest TensorFlow wheel compatibility.

Quickstart — run the Flask demo locally (browser camera + upload UI)
1. Create and activate a Python virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Start the Flask server (serves the interactive web demo at http://127.0.0.1:5000):

```powershell
python app.py --flask
```

3. Open your browser at http://127.0.0.1:5000

Features of the Flask web UI
- Tabbed Camera / Upload interface. Camera mode streams webcam frames and shows live overlay predictions. Upload mode supports drag-and-drop or file selection and shows a preview + one-shot predictions.
- Pending indicator (spinner) while the server processes a frame.
- Recent prediction history (last 10 predictions) and a Clear History button.

Quickstart — run the Streamlit demo
The same `app.py` contains the Streamlit demo. To run it via Streamlit locally:

```powershell
streamlit run app.py -- --streamlit
```

Streamlit mode provides an image uploader and optional webcam demo (requires `streamlit-webrtc`). Streamlit is convenient for sharing on Streamlit Cloud.

Model availability and runtime downloader
- By default the app expects a Keras model at `models/sign_language_model.h5` and labels at `models/labels.json`.
- To avoid committing large models to git, you can host the model in cloud storage (S3, GCS, or any public URL) and set the environment variable `MODEL_URL` before starting the app. `app.py` will try to download the model at startup if `models/sign_language_model.h5` is missing.

Example (PowerShell) to set MODEL_URL and start Streamlit:

```powershell
$env:MODEL_URL = 'https://your-bucket/path/sign_language_model.h5'
streamlit run app.py -- --streamlit
```

Or start Flask with the same environment variable:

```powershell
$env:MODEL_URL = 'https://your-bucket/path/sign_language_model.h5'
python app.py --flask
```

Notes on hosting the model
- Recommended: host models on cloud storage and keep `models/` in `.gitignore`.
- Optionally use Git LFS for models, but Streamlit Cloud and some CI systems have limits — prefer external hosting.

Troubleshooting
- "Failed to fetch" in the browser when calling `/predict`:
	- Ensure the Flask server is running (see the terminal where you started `python app.py --flask`).
	- Check the Flask terminal for tracebacks. Common causes: missing TensorFlow, missing model file, or an exception when decoding images.
	- If your page is served from a different origin, CORS may be required. For quick local testing you can enable CORS in `app.py` (install `flask-cors` and call `CORS(flask_app)` after creating the app).
	- If serving over HTTPS, avoid mixing insecure HTTP `http://127.0.0.1:5000` requests from a secure page (browser will block mixed content).

- If the model fails to load at startup:
	- Confirm the files are at `models/sign_language_model.h5` and `models/labels.json`.
	- Check Python/TensorFlow compatibility (Python 3.10/3.11 recommended).

Development notes and recommended next steps
- Add your dataset to `data/` and run `prepare_data.py` to create `data/train` and `data/val`.
- Train the model with `train.py` and save to `models/sign_language_model.h5`.
- Evaluate using `evaluate.py`.
- The repo includes both the Flask frontend and Streamlit demo. If you only want Streamlit, you can delete `templates/` and `static/` and make `app.py` the Streamlit entrypoint.

Deployment (Streamlit Cloud)
1. Push your repository to GitHub.
2. On share.streamlit.io create a new app and link it to your repo & branch.
3. Set the environment variable `MODEL_URL` in the Streamlit app settings (or upload the model to your repo via Git LFS, though external hosting is preferred).

Useful commands

Start Flask (dev):
```powershell
python app.py --flask
```

Start Streamlit locally:
```powershell
streamlit run app.py -- --streamlit
```

Add CORS (quick local test):
```powershell
pip install flask-cors
```
Then in `app.py` add:
```python
from flask_cors import CORS
flask_app = Flask(...)
CORS(flask_app)
```

Contact / next steps I can help with
- I can add a small `models/README.md` explaining how to host models on S3/GCS and generate a secure URL.
- I can add additional UI polish (Bootstrap/CSS), improve mobile support, or add server-side batching for faster throughput.

---

Thanks — try running `python app.py --flask` and open http://127.0.0.1:5000 to see the interactive Camera/Upload demo.
