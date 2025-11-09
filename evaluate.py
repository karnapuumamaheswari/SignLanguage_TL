"""evaluate.py

Evaluate a trained model on the validation dataset and produce a classification report
and confusion matrix. Saves reports to the working directory.

Usage:
    python evaluate.py --model models\sign_language_model.h5 --labels models\labels.json --data_dir .\data\val --batch_size 32

"""
import argparse
import json
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def main(model_path, labels_path, data_dir, img_size=224, batch_size=32):
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    idx_to_class = {int(k): v for k, v in labels.items()}

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    model = load_model(model_path)

    steps = int(np.ceil(generator.samples / batch_size))
    preds = model.predict(generator, steps=steps, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = generator.classes

    target_names = [generator.class_indices[k] for k in sorted(generator.class_indices, key=lambda x: generator.class_indices[x])]

    # classification report
    report = classification_report(y_true, y_pred, target_names=sorted(generator.class_indices.keys()))
    print(report)
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    print('Saved classification_report.txt')

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(cm, index=sorted(generator.class_indices.keys()), columns=sorted(generator.class_indices.keys()))
    df.to_csv('confusion_matrix.csv')
    print('Saved confusion_matrix.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to trained Keras model (.h5)')
    parser.add_argument('--labels', required=True, help='Path to labels.json mapping indices to class names')
    parser.add_argument('--data_dir', required=True, help='Path to validation data directory (root with class subfolders)')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    main(args.model, args.labels, args.data_dir, args.img_size, args.batch_size)
