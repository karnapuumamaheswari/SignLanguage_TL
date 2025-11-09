<<<<<<< HEAD
# Sign Language Alphabet Recognition using Transfer Learning

This project demonstrates how to use Transfer Learning (MobileNetV2) to classify American Sign Language (ASL) alphabets (A-Z) and run real-time predictions using a webcam.

## Contents
- `train.py` - Training script that builds and trains a MobileNetV2-based classifier.
- `prepare_data.py` - Helper to split the Kaggle ASL dataset into `train` / `val` folders.
- `realtime.py` - Real-time webcam demo using the trained model.
- `requirements.txt` - Python dependencies.

## Setup (Windows PowerShell)
1. Create a virtual environment and activate it:

```powershell
python -m venv .venv
# SignLanguage_TL — ASL Alphabet Recognition (MobileNetV2 transfer learning)

This repository provides scripts and a web demo for recognizing the American Sign Language (ASL) alphabet (A–Z) using transfer learning (MobileNetV2 + custom head). It includes training, evaluation, a desktop realtime demo, and a browser frontend that captures webcam frames and sends them to a Flask server for inference.

---

## What's new (recent changes)
- The web demo client (`static/main.js`) now prevents overlapping POSTs: the client waits for an outstanding `/predict` response before sending the next frame. This reduces request pile-up when the server is slower than the client interval.
- The Flask server (`app.py`) performs a short model warmup at startup (dummy prediction) and uses `predict_on_batch` to reduce per-request overhead. These changes smooth the first-request latency and slightly lower runtime overhead.
- A simple `GET /favicon.ico` route was added to avoid spamming 404s in server logs.

---

## Repository layout
- `train.py` — training script (MobileNetV2 base, custom head). Saves model and `labels.json`.
- `prepare_data.py` — split/prepare your raw dataset into `data/train` and `data/val`.
- `evaluate.py` — evaluate model on validation set; produces classification report and confusion matrix.
- `predict_webcam.py` / `realtime.py` — OpenCV-based realtime demo (desktop).
- `app.py` — Flask web server providing `/` (UI) and `/predict` (POST image dataURL → JSON prediction).
- `templates/index.html` and `static/main.js` — browser UI and client logic.
- `models/` — trained models and `labels.json` (not included by default; recommended to ignore in git).

---

## Setup (Windows PowerShell)
1. Create and activate a virtual environment (recommended Python 3.10 or 3.11 — TensorFlow wheels may not be available for newer releases):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

If `tensorflow` fails to install under your Python version, create a conda environment with Python 3.10/3.11 or install a compatible interpreter.

---

## Prepare dataset
1. Download the "ASL Alphabet" dataset (for example from Kaggle).
2. Unzip to a folder containing 26 subfolders (A–Z).
3. Run the helper to create `data/train` and `data/val`:

```powershell
python prepare_data.py --src "C:\path\to\asl_alphabet_train" --dest .\data --val_split 0.2
```

This will create `./data/train/<class>` and `./data/val/<class>`.

---

## Train
Train with:

```powershell
python train.py --data_dir .\data --epochs 12 --batch_size 32 --output models\sign_language_model.h5
```

The script saves the model and a `labels.json` mapping next to the model (e.g. `models\labels.json`).

---

## Web demo (browser)
1. Start the Flask server (ensure your venv with TensorFlow is active):

```powershell
# activate venv if needed
.\.venv\Scripts\Activate.ps1
python app.py
```

2. Open the UI at: http://127.0.0.1:5000

3. Controls and notes:
- Click Start to grant webcam access and begin sending frames. The UI posts frames (center-cropped 224×224 JPEG) to `/predict` at the interval set in the "Interval (ms)" input.
- Default interval is 250 ms (~4 FPS). If your CPU or server is slower, increase this interval.
- The client prevents overlapping POSTs: it won't send a new request while a previous `/predict` is still pending. This avoids queuing many requests on the server.

4. Server-side behaviour and what changed:
- The server now warms the model on startup (runs a dummy prediction) to reduce the first-request latency spike.
- The `/predict` endpoint uses `predict_on_batch` for single-batch inference which has slightly lower overhead on many TF/Keras versions.
- A simple `GET /favicon.ico` route returns 204 to avoid 404 log entries.

If you change `static/main.js`, make sure the script is loaded after the DOM (the template already includes the script at the end of the body). If you ever see "Cannot read property ... of null" in the browser console, reload the page and verify the script tag is present once (no duplicates).

---

## Realtime desktop demo
Use the OpenCV-based demo if you prefer a local window instead of the browser:

```powershell
python predict_webcam.py --model models\sign_language_model.h5 --labels models\labels.json
```

Press `q` in the display window to quit.

---

## Evaluate
To evaluate on the validation set:

```powershell
python evaluate.py --model models\sign_language_model.h5 --labels models\labels.json --data_dir .\data\val --batch_size 32
```

The script writes `classification_report.txt` and `confusion_matrix.csv`.

---

## Troubleshooting
- Browser console errors about missing elements: ensure the UI is served from `app.py` and that `templates/index.html` includes `<script src="/static/main.js"></script>` once at the end of the body.
- If `/predict` is slow or you see many overlapping requests in server logs, increase the client interval (ms) in the UI or use the stop/start controls. The client now avoids overlapping requests, which should reduce load.
- If TensorFlow installation fails on Windows, verify your Python version (use 3.10/3.11) or use conda to create a compatible environment.

---

## Next steps and optional improvements
- Add server-side rate limiting (Flask-Limiter) to protect the server from many clients.
- Convert the model to TF.js to run inference directly in the browser (removes server inference cost).
- Expose a model-upload endpoint so non-developers can swap models without restarting the server.
- Improve UI: show a pending/pulse indicator while waiting for `/predict`, display effective FPS, and show a small latency histogram.

---

If you'd like, I can add a small "request pending" indicator to the web UI and lower the default interval, or add server-side rate limiting — tell me which and I'll implement it.
