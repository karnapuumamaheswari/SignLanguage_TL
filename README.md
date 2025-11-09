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
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Prepare dataset
1. Download the "ASL Alphabet Dataset" from Kaggle (https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
2. Unzip the dataset; you'll get a directory with 26 class folders (A-Z).
3. Use `prepare_data.py` to split the images into `data/train` and `data/val`:

```powershell
python prepare_data.py --src "C:\path\to\asl_alphabet_train" --dest "./data" --val_split 0.2
```

This will create `./data/train/<class>` and `./data/val/<class>` folders.

## Train
Train the model with:

```powershell
python train.py --data_dir .\data --epochs 12 --batch_size 32 --output models\sign_language_model.h5
```

The training script saves the model and a `labels.json` mapping next to the model (for example `models\labels.json`).

## Real-time demo
After training, run the webcam demo. `predict_webcam.py` will locate `labels.json` next to the model if you omit `--labels`.

```powershell
python predict_webcam.py --model models\sign_language_model.h5 --labels models\labels.json
```

Or omit `--labels` if `labels.json` is next to the model:

```powershell
python predict_webcam.py --model models\sign_language_model.h5
```

Press `q` in the display window to quit.

## Evaluate
You can evaluate the trained model on the validation set with `evaluate.py` which produces a classification report and a confusion matrix CSV:

```powershell
python evaluate.py --model models\sign_language_model.h5 --labels models\labels.json --data_dir .\data\val --batch_size 32
```

The script writes `classification_report.txt` and `confusion_matrix.csv` to the working directory.

## Notes
- Use good lighting and a plain background for best results.
- Reduce `batch_size` if you run out of memory.
- For better accuracy, unfreeze some MobileNetV2 layers and fine-tune with a lower learning rate.

