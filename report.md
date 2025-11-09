# Sign Language Alphabet Recognition using Transfer Learning

## Project Title
Sign Language Alphabet Recognition using Transfer Learning

## Objective
Develop a deep learning model to recognize hand gestures representing English alphabets (A–Z) in American Sign Language (ASL) using Transfer Learning. The system should perform accurate, real-time predictions from a webcam feed.

## Project Structure
```
SignLanguage_TL/
├── data/                # train/val split folders (A-Z)
├── models/              # store model and labels.json
├── train.py             # training script
├── predict_webcam.py    # real-time webcam demo
├── realtime.py          # internal real-time demo implementation
├── prepare_data.py      # split raw dataset into train/val
├── requirements.txt     # project dependencies
├── README.md            # project overview and instructions
└── report.md            # this document (presentation-ready)
```

## Dataset Details
- Dataset: ASL Alphabet Dataset (Kaggle)
- Classes: 26 (A–Z)
- Images per class: ~3000
- Total images: ~87,000
- Image size: 200×200 (original), resized to 224×224 for training
- Split: 80% train / 20% validation

## Technology Stack
- Python 3.8+
- TensorFlow / Keras
- MobileNetV2 (pre-trained on ImageNet)
- OpenCV (real-time webcam capture)
- NumPy, Matplotlib, scikit-learn

## Module Explanations

### prepare_data.py
- Purpose: Split original dataset (one folder per class) into `data/train/` and `data/val/`.
- Inputs: `--src` path to original folder, `--dest` output folder, `--val_split` fraction for validation.
- Behavior: Copies images into train/val while preserving class folders; uses a random seed for reproducibility.

### train.py
- Purpose: Train a classifier using MobileNetV2 as a frozen feature extractor and a custom head for 26-class classification.
- Key steps:
  1. Create ImageDataGenerator with preprocessing and augmentation for the training set.
  2. Build model: MobileNetV2 (include_top=False) -> GlobalAveragePooling2D -> Dense(512, relu) -> Dropout -> Dense(26, softmax).
  3. Compile with Adam, categorical_crossentropy.
  4. Use callbacks (ModelCheckpoint, ReduceLROnPlateau, EarlyStopping).
  5. Save best model to the provided output path and write `labels.json` next to the model.

### realtime.py / predict_webcam.py
- Purpose: Real-time inference using a webcam feed.
- Steps:
  1. Load model and labels.
  2. Capture frame, extract a square ROI, resize to 224×224, preprocess.
  3. Run model prediction, map index to class label, overlay label+confidence on frame.
  4. Display FPS and allow exit via `q`.

## Workflow (Step-by-step)
1. Download dataset from Kaggle and extract.
2. Prepare dataset:
   ```
   python prepare_data.py --src "C:\path\to\asl_alphabet_train" --dest .\data --val_split 0.2
   ```
3. Train the model:
   ```
   python train.py --data_dir .\data --epochs 12 --batch_size 32 --output models\sign_language_model.h5
   ```
4. Run real-time demo:
   ```
   python predict_webcam.py --model models\sign_language_model.h5 --labels models\labels.json
   ```

## Evaluation Metrics
- Accuracy (train/val)
- Confusion matrix
- Classification report (precision, recall, f1-score)

## Expected Results
- Training accuracy: ~95–97%
- Validation accuracy: ~90–94%
- Loss: < 0.3

## Improvements and Future Work
- Fine-tune top MobileNetV2 layers for better performance.
- Use additional augmentations (brightness, contrast, cutout).
- Move to sequence models for word recognition.
- Deploy with TensorFlow Lite for mobile apps.

## References
- Kaggle — ASL Alphabet Dataset
- TensorFlow / Keras documentation
- OpenCV documentation

---

This document is ready to paste into your report or use as presentation notes; edit any part to match your experimental numbers and team roles.
