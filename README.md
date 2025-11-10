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
<!-- 
<!-- ```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
``` --> -->

2. Place your trained Keras model and labels next to each other:

- `models/sign_language_model.h5`
- `models/labels.json`

Run Streamlit locally

```powershell
streamlit run streamlit_app.py
```

Open the URL printed by Streamlit (usually `http://localhost:8501`). The Streamlit app supports:
- Image upload prediction (works out of the box).
- Optional webcam demo (in-browser) if you install `streamlit-webrtc`.

Deploying to Streamlit Cloud (simple)
1. Push your repository to GitHub (include `streamlit_app.py` and `requirements.txt` at the repo root).
2. Create a new app on share.streamlit.io and point it at your repo + branch. Set the entrypoint to `streamlit_app.py` (default is fine).

Notes for cloud deployment
- Keep models reasonably small. Large `.h5` models may cause long startup times; consider saving a smaller/frozen/quantized model if needed.
- Alternatively, host the model in cloud storage (S3, GCS) and download at startup.

Optional: Live webcam with `streamlit-webrtc`
- Install `streamlit-webrtc` (already in `requirements.txt` as optional). The Streamlit app will automatically enable the webcam demo if the package is available.
- Browser webcam support in Streamlit uses WebRTC and runs the model server-side. For heavy loads, consider running a dedicated inference server.

Folder / file recommendations for a Streamlit-friendly repo
- Keep `streamlit_app.py` and `requirements.txt` at the repository root — Streamlit Cloud expects these.
- Keep `models/` at the root (git-ignored) or add a `models/README.md` explaining how to obtain the model.
- Keep `data/` and training scripts in the repo, but do not commit large datasets.

Run-time notes
- Streamlit runs the app in a long-lived process. Loading and warming the model at import/startup (the app does this) avoids latency on the first request.
- Make sure your Python runtime on the host supports TensorFlow; Streamlit Cloud runs standard Linux images and supports TF in many versions, but verify compatibility.

If you prefer, I can:
- Add a small `models/README.md` that explains where to place/download models and how to convert to TF.js/SavedModel.
- Replace the Flask demo entirely with Streamlit assets and remove `templates/`/`static/` to simplify the repo.
- Add a small UI improvement: a visible "request pending" indicator and an FPS/latency readout in the Streamlit app.

Next steps I can take for you
- Add the pending indicator and FPS readout inside `streamlit_app.py` (quick change).
- Add a sample `Procfile` for other hosts (if you plan to deploy on Heroku-like environments).
- Show how to host the model in cloud storage and download it at startup (helpful if model is large).

---

Enjoy — let me know which option you want next (pending indicator, remove Flask demo, or cloud model hosting).
Produce a classification report and confusion matrix:
