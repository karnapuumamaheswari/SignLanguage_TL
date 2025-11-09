"""prepare_data.py

Split a dataset where each class is a folder of images into train/val folders.

Usage:
    python prepare_data.py --src PATH_TO_ORIG_FOLDER --dest ./data --val_split 0.2

"""
import os
import argparse
import shutil
import random
from pathlib import Path


def split_dataset(src_dir, dest_dir, val_split=0.2, seed=42):
    random.seed(seed)
    src = Path(src_dir)
    dest = Path(dest_dir)
    train_dir = dest / "train"
    val_dir = dest / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    class_folders = [p for p in src.iterdir() if p.is_dir()]
    if not class_folders:
        raise RuntimeError(f"No class folders found in {src}")

    for c in class_folders:
        images = [p for p in c.iterdir() if p.is_file()]
        random.shuffle(images)
        n_val = int(len(images) * val_split)
        val_images = images[:n_val]
        train_images = images[n_val:]

        (train_dir / c.name).mkdir(parents=True, exist_ok=True)
        (val_dir / c.name).mkdir(parents=True, exist_ok=True)

        for p in train_images:
            shutil.copy(p, train_dir / c.name / p.name)
        for p in val_images:
            shutil.copy(p, val_dir / c.name / p.name)

    print(f"Finished splitting. Train: {train_dir}, Val: {val_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='Path to original dataset (folders A-Z)')
    parser.add_argument('--dest', default='./data', help='Destination folder (will contain train/val)')
    parser.add_argument('--val_split', type=float, default=0.2)
    args = parser.parse_args()
    split_dataset(args.src, args.dest, args.val_split)
