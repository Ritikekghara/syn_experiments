"""Split data into train/val/test folders.
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "data"
DEST_DIR = ROOT / "data_split"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> list[Path]:
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_files(files: Iterable[Path], target_dir: Path) -> None:
    ensure_dir(target_dir)
    for f in files:
        shutil.copy2(f, target_dir / f.name)


def split_list(items: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    random.shuffle(items)
    n = len(items)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val
    train_items = items[:n_train]
    val_items = items[n_train : n_train + n_val]
    test_items = items[n_train + n_val : n_train + n_val + n_test]
    return train_items, val_items, test_items


def main() -> None:
    if not SRC_DIR.exists():
        raise SystemExit(f"Source folder not found: {SRC_DIR}")

    random.seed(SEED)

    classes = [p for p in SRC_DIR.iterdir() if p.is_dir()]
    if not classes:
        raise SystemExit(f"No class folders found in {SRC_DIR}")

    print(f"Found classes: {[c.name for c in classes]}")
    print(f"Writing split to: {DEST_DIR}")

    for class_dir in classes:
        images = list_images(class_dir)
        if not images:
            print(f"Skipping empty class folder: {class_dir}")
            continue

        train_items, val_items, test_items = split_list(images)

        copy_files(train_items, DEST_DIR / "train" / class_dir.name)
        copy_files(val_items, DEST_DIR / "val" / class_dir.name)
        copy_files(test_items, DEST_DIR / "test" / class_dir.name)

        print(
            f"{class_dir.name}: total={len(images)} "
            f"train={len(train_items)} val={len(val_items)} test={len(test_items)}"
        )


if __name__ == "__main__":
    main()
