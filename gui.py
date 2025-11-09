"""Simple GUI for real-time ASL alphabet prediction.

Usage:
    python gui.py --model models\sign_language_model.h5 --labels models\labels.json

This creates a Tkinter window showing the webcam feed, a square ROI, and the predicted
alphabet with confidence. You can load a model/labels via the "Load Model" button.
"""
import argparse
import json
import threading
import time
import os
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    from PIL import Image, ImageTk
except Exception:
    raise SystemExit('Tkinter and Pillow are required for the GUI. Install Pillow and use a Python build that includes Tkinter.')

import cv2
import numpy as np

MODEL_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.models import load_model
except Exception:
    MODEL_AVAILABLE = False


class ASLApp:
    def __init__(self, root, model_path=None, labels_path=None, img_size=224):
        self.root = root
        self.root.title('ASL Alphabet - Live Demo')
        self.img_size = img_size

        # Video frame
        self.video_label = tk.Label(root)
        self.video_label.pack()

        # Prediction label
        self.pred_var = tk.StringVar(value='Model not loaded')
        self.pred_label = tk.Label(root, textvariable=self.pred_var, font=('Helvetica', 18))
        self.pred_label.pack(pady=8)

        # Controls frame
        ctrl = tk.Frame(root)
        ctrl.pack()

        self.load_btn = tk.Button(ctrl, text='Load Model', command=self.load_model_dialog)
        self.load_btn.grid(row=0, column=0, padx=4)

        self.start_btn = tk.Button(ctrl, text='Start', command=self.start, state='normal')
        self.start_btn.grid(row=0, column=1, padx=4)
        self.stop_btn = tk.Button(ctrl, text='Stop', command=self.stop, state='disabled')
        self.stop_btn.grid(row=0, column=2, padx=4)

        # Status
        self.status_var = tk.StringVar(value='Ready')
        tk.Label(root, textvariable=self.status_var).pack(pady=4)

        # Model and labels
        self.model = None
        self.labels = None
        if model_path and labels_path and os.path.exists(model_path) and os.path.exists(labels_path):
            try:
                self.model = load_model(model_path)
                with open(labels_path, 'r') as f:
                    self.labels = {int(k): v for k, v in json.load(f).items()}
                self.pred_var.set('Model loaded')
            except Exception as e:
                self.pred_var.set('Failed to load model')
                print('Model load error:', e)

        # Camera
        self.cap = None
        self.running = False
        self._frame = None
        self._lock = threading.Lock()

        # Start update loop
        self.update_ui()

    def load_model_dialog(self):
        model_file = filedialog.askopenfilename(title='Select model (.h5)', filetypes=[('Keras model', '*.h5'), ('All files', '*.*')])
        if not model_file:
            return
        labels_file = filedialog.askopenfilename(title='Select labels.json', filetypes=[('JSON', '*.json'), ('All files', '*.*')])
        if not labels_file:
            messagebox.showwarning('Labels missing', 'Please select a labels.json file after selecting the model')
            return

        try:
            self.status_var.set('Loading model...')
            self.root.update()
            self.model = load_model(model_file)
            with open(labels_file, 'r') as f:
                self.labels = {int(k): v for k, v in json.load(f).items()}
            self.pred_var.set('Model loaded')
            self.status_var.set('Model loaded')
        except Exception as e:
            messagebox.showerror('Load error', f'Failed to load model/labels: {e}')
            self.status_var.set('Load failed')

    def start(self):
        if not MODEL_AVAILABLE and self.model is None:
            messagebox.showerror('TensorFlow missing', 'TensorFlow is not available in this environment. Install TensorFlow to use the model.')
            return

        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror('Camera error', 'Cannot open webcam')
                self.cap = None
                return

        self.running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_var.set('Running')
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_var.set('Stopped')
        # release handled in capture loop

    def _capture_loop(self):
        fps_time = time.time()
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            side = min(w, h)
            cx, cy = w // 2, h // 2
            x1 = cx - side // 2
            y1 = cy - side // 2
            x2 = x1 + side
            y2 = y1 + side

            roi = frame[y1:y2, x1:x2]
            # draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # prediction
            label = '?'
            conf = 0.0
            if self.model is not None:
                try:
                    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    arr = np.array(img).astype('float32')
                    arr = preprocess_input(arr)
                    arr = np.expand_dims(arr, axis=0)
                    preds = self.model.predict(arr)
                    idx = int(np.argmax(preds[0]))
                    conf = float(np.max(preds[0]))
                    label = self.labels.get(idx, '?') if self.labels else str(idx)
                except Exception as e:
                    # prediction error; just continue
                    print('Prediction error:', e)

            # overlay
            text = f"{label} ({conf*100:.1f}%)"
            cv2.putText(frame, text, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

            # fps
            now = time.time()
            fps = 1.0 / (now - fps_time) if (now - fps_time) > 0 else 0.0
            fps_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # convert for Tkinter
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img_rgb)
            with self._lock:
                self._frame = ImageTk.PhotoImage(im_pil)

            time.sleep(0.02)

        # cleanup
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_var.set('Idle')

    def update_ui(self):
        # update video label
        with self._lock:
            if self._frame is not None:
                self.video_label.configure(image=self._frame)
                self.video_label.image = self._frame
        self.root.after(30, self.update_ui)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to trained Keras model (.h5)')
    parser.add_argument('--labels', help='Path to labels.json')
    args = parser.parse_args()

    root = tk.Tk()
    app = ASLApp(root, model_path=args.model, labels_path=args.labels)
    root.protocol('WM_DELETE_WINDOW', root.quit)
    root.mainloop()


if __name__ == '__main__':
    main()
