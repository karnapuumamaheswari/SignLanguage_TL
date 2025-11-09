import cv2
import tensorflow as tf
import numpy as np
import json
import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def preprocess_frame(frame):
    """Resize and normalize frame for model input."""
    if frame is None or frame.size == 0:
        return None
    img = cv2.resize(frame, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def main(model_path, labels_path, camera_index=0):
    print("🔍 Loading model and labels...")
    model = tf.keras.models.load_model(model_path)

    # Load label mappings
    with open(labels_path) as f:
        labels = json.load(f)

    # Detect label format and normalize it
    # (handle both {"A": 0} and {"0": "A"} formats)
    first_key = list(labels.keys())[0]
    if first_key.isdigit():
        # Convert if keys are numbers
        labels = {v: int(k) for k, v in labels.items()}

    labels_list = list(labels.keys())
    print(f"✅ Loaded {len(labels_list)} labels:", labels_list)

    print("🎥 Starting webcam... (Press 'Q' to stop and see final result)")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Error: Cannot access webcam.")
        return

    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Mirror the frame for natural interaction
        frame = cv2.flip(frame, 1)
        input_data = preprocess_frame(frame)
        if input_data is None:
            continue

        # Run prediction
        preds = model.predict(input_data, verbose=0)
        pred_idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        predictions.append(pred_idx)

        # Convert numeric prediction → label (A, B, C, etc.)
        label = labels_list[pred_idx]

        # Display result on screen
        cv2.putText(frame, f"Prediction: {label}",
                    (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, f"Confidence: {confidence * 100:.2f}%",
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Show webcam window
        cv2.imshow("🖐 Sign Language Detection (Press 'Q' to Quit)", frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Determine final prediction
    if predictions:
        final_idx = max(set(predictions), key=predictions.count)
        final_label = labels_list[final_idx]
        print("\n✅ Final Prediction:")
        if final_label == "space":
            print("➡️ You signed: Space Gesture")
        elif final_label == "nothing":
            print("➡️ You signed: No Sign Detected")
        elif final_label == "del":
            print("➡️ You signed: Delete / Backspace Gesture")
        else:
            print(f"➡️ You signed the letter: '{final_label}' (in ASL)")
    else:
        print("\n⚠️ No prediction made.")

    print("🛑 Webcam closed successfully.")
