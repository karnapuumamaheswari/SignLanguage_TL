<<<<<<< HEAD
# Sign Language Alphabet Recognition using Transfer Learning

This project demonstrates how to use Transfer Learning (MobileNetV2) to classify American Sign Language (ASL) alphabets (A-Z) and run real-time predictions using a webcam.

## Contents
- `train.py` - Training script that builds and trains a MobileNetV2-based classifier.
- `prepare_data.py` - Helper to split the Kaggle ASL dataset into `train` / `val` folders.
- `realtime.py` - Real-time webcam demo using the trained model.
- `requirements.txt` - Python dependencies.
## SignLanguage_TL — ASL Alphabet Recognition

Inspiring people and machines to communicate.

This project shows how to build a compact, practical ASL alphabet recognizer using transfer learning (MobileNetV2). It includes training utilities, evaluation scripts, a desktop demo, and a friendly browser-based demo so you can try real-time recognition in seconds.

Why this project exists
- Make sign language technology approachable for hobbyists and researchers.
- Provide a clear, practical example of transfer learning and real-time inference.
- Offer both a desktop and browser demo so you can test locally or in the browser + server.

Key highlights
- Lightweight MobileNetV2 base with a small custom head for fast inference.
- Browser demo that captures webcam frames and posts them to a Flask server for prediction.
- Robust client: prevents overlapping requests to avoid server overload.
- Server warms the model at startup to smooth the first request latency.

Getting started (quick)
1. Clone the repo and open it in PowerShell.
2. Create a virtual environment (Python 3.10/3.11 recommended) and activate it:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

Prepare the dataset
1. Download an ASL alphabet dataset (for example, the Kaggle "ASL Alphabet" dataset).
2. Unzip so you have a folder with 26 class subfolders (A–Z).
3. Run the helper to split into train/validation:

```powershell
python prepare_data.py --src "C:\path\to\asl_alphabet_train" --dest .\data --val_split 0.2
```

Train
Train a model using transfer learning:

```powershell
python train.py --data_dir .\data --epochs 12 --batch_size 32 --output models\sign_language_model.h5
```

The training script saves the model and a `labels.json` mapping near the model (e.g. `models\labels.json`).

Browser demo (fast test)
1. Start the server (activate venv if needed):

<!-- ```powershell
.\.venv\Scripts\Activate.ps1
python app.py
``` -->

2. Open http://127.0.0.1:5000 in your browser. Click "Start" to grant webcam access and begin.

Notes
- The client sends center-cropped 224×224 JPEG frames to `/predict`. Default interval: 250 ms (~4 FPS).
- The client will not send a new request until the previous one completes (prevents request pile-up).
- The server warms the model on startup and uses `predict_on_batch` for single-batch inference.

Desktop demo
Use the OpenCV-based demo if you prefer a local window:

```powershell
python predict_webcam.py --model models\sign_language_model.h5 --labels models\labels.json
```

Evaluate
Produce a classification report and confusion matrix:

```powershell
python evaluate.py --model models\sign_language_model.h5 --labels models\labels.json --data_dir .\data\val --batch_size 32
```

Troubleshooting & tips
- If you see browser console errors about missing elements, ensure the HTML template includes `static/main.js` once at the end of the body. The client script is wrapped in DOMContentLoaded to avoid early access errors.
- If TensorFlow fails to install on your Python version, switch to Python 3.10/3.11 or create a conda environment.
- Increase the UI interval (ms) if your machine or server is slow.

What's next (ideas)
- Add server-side rate limiting (Flask-Limiter) to protect the server when many clients connect.
- Convert the model to TF.js so inference can run entirely in the browser.
- Add a visual "request pending" indicator and an FPS/latency readout in the UI.

Contributing
- Contributions, issues and feature requests are welcome. Please open a GitHub issue or a pull request.

License
- This project is provided for learning and experimentation. Check `LICENSE` if present or add one for redistribution.

Enjoy — build something that helps people! 🚀


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
