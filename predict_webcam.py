"""predict_webcam.py

Alias script for real-time webcam prediction. Calls the same logic as `realtime.py`.

Usage:
    python predict_webcam.py --model models\sign_language_model.h5 --labels models\labels.json

"""
import argparse
import os
from realtime import main as run_realtime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to trained Keras model (.h5)')
    parser.add_argument('--labels', required=False, help='Path to labels.json mapping indices to class names (if omitted, will look next to model)')
    parser.add_argument('--camera', type=int, default=0, help='Camera device id (default 0)')
    args = parser.parse_args()

    labels_path = args.labels
    if not labels_path:
        # try to find labels.json next to model
        model_dir = os.path.dirname(args.model)
        candidate = os.path.join(model_dir, 'labels.json')
        if os.path.exists(candidate):
            labels_path = candidate
        else:
            raise SystemExit('labels.json not provided and not found next to model. Provide --labels')

    run_realtime(args.model, labels_path, args.camera)
