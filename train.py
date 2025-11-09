"""train.py

Train a MobileNetV2-based classifier on the ASL alphabet dataset.
Saves the trained model and a labels.json mapping file.

Usage:
    python train.py --data_dir ./data --epochs 12 --batch_size 32 --output sign_language_model.h5
"""
import os
import json
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def build_model(input_shape=(224,224,3), num_classes=26, dropout_rate=0.5):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model


def main(data_dir, img_size=224, batch_size=32, epochs=12, output='sign_language_model.h5'):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train') if os.path.isdir(os.path.join(data_dir, 'train')) else data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training' if os.path.isdir(os.path.join(data_dir, 'train')) else None,
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'val') if os.path.isdir(os.path.join(data_dir, 'val')) else data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation' if os.path.isdir(os.path.join(data_dir, 'val')) else None,
        shuffle=False
    )

    # If user provided a root folder (with train/val subfolders), the flow_from_directory calls above
    # will already point to correct folders. If not, using subset may be None and flow_from_directory
    # will treat data_dir as the directory containing class subfolders.

    num_classes = len(train_generator.class_indices)
    print(f"Found {num_classes} classes: {train_generator.class_indices}")

    model = build_model(input_shape=(img_size, img_size, 3), num_classes=num_classes)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    ckpt_cb = callbacks.ModelCheckpoint(output, monitor='val_accuracy', save_best_only=True, verbose=1)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[ckpt_cb, reduce_lr, early_stop]
    )

    # Save final model (redundant if checkpoint saved best)
    # Ensure output directory exists, then save model
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    model.save(output)
    print(f"Model saved to {output}")

    # Save labels mapping next to the model
    labels = {v: k for k, v in train_generator.class_indices.items()}
    labels_path = os.path.join(out_dir if out_dir else '.', 'labels.json')
    with open(labels_path, 'w') as f:
        json.dump(labels, f)
    print(f'Saved labels to {labels_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data', help='Path to training data (train/val folders or root with class folders)')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--output', default='sign_language_model.h5')
    args = parser.parse_args()
    main(args.data_dir, args.img_size, args.batch_size, args.epochs, args.output)
