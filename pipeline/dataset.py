"""
pipeline/dataset.py
====================
Downloads Food-101 and builds two dataset folders:

  data/carb_dataset/train_eval/ - carb range labels for Food-101 TRAIN split (model training)
  data/carb_dataset/final_eval/ - carb range labels for Food-101 TEST split  (evaluation only)

Keeping train_eval and final_eval separate prevents data leakage:
the model trains on train_eval and is evaluated on the unseen final_eval.
"""

import json
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DATA_DIR, FOOD101_DIR, DATASET_DIR,
    FOOD_BASE_CARBS, PORTION_MULTIPLIER,
    IMAGES_PER_CLASS, grams_to_range,
)

FOOD101_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"


# ──────────────────────────────────────────────
# DOWNLOAD
# ──────────────────────────────────────────────

def download_food101():
    dest        = Path(DATA_DIR) / "food101"
    extract_dir = dest / "food-101"
    dest.mkdir(parents=True, exist_ok=True)

    if extract_dir.exists():
        print("OK Food-101 already downloaded.")
        return

    tar_path = dest / "food-101.tar.gz"
    print(f"Downloading Food-101 (~5 GB) to {tar_path}")
    print("  This takes 10-30 minutes depending on your connection.")

    def _hook(count, block, total):
        pct = min(int(count * block * 100 / total), 100)
        bar = "=" * (pct // 5)
        print(f"\r  {pct:3d}%  {bar:<20}", end="", flush=True)

    urllib.request.urlretrieve(FOOD101_URL, str(tar_path), reporthook=_hook)
    print("\n  Extracting...")
    with tarfile.open(str(tar_path), "r:gz") as tar:
        tar.extractall(str(dest))
    tar_path.unlink()
    print("OK Extraction complete.")


# ──────────────────────────────────────────────
# PORTION SIZE HEURISTIC
# ──────────────────────────────────────────────

def estimate_portion(img_path: Path) -> int:
    """
    Estimate portion size from image (0=small, 1=medium, 2=large).
    Compares brightness of central crop vs full image.
    Foods filling more of the frame score higher = larger portion.
    Documented as a synthetic labelling method in dissertation methodology.
    """
    try:
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32)
        full_mean   = arr.mean()
        h, w        = arr.shape[:2]
        cy, cx      = h // 4, w // 4
        centre_mean = arr[cy:h-cy, cx:w-cx].mean()
        ratio       = centre_mean / (full_mean + 1e-6)
        if ratio < 0.95:   return 0
        elif ratio < 1.05: return 1
        else:              return 2
    except Exception:
        return 1  # default to medium


# ──────────────────────────────────────────────
# BUILD DATASETS
# ──────────────────────────────────────────────

def build_datasets():
    images_dir = Path(FOOD101_DIR) / "images"
    meta_dir   = Path(FOOD101_DIR) / "meta"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Food-101 not found at {images_dir}. Run download_food101() first."
        )

    with open(meta_dir / "train.txt") as f:
        train_keys = set(line.strip() for line in f)

    train_dir = Path(DATASET_DIR) / "train_eval"
    eval_dir  = Path(DATASET_DIR) / "final_eval"

    stats = {
        "train": {str(r): 0 for r in range(5)},
        "final": {str(r): 0 for r in range(5)},
    }
    records  = []

    print(f"\nBuilding datasets (up to {IMAGES_PER_CLASS} images per class)...")

    for food_class, base_carbs in FOOD_BASE_CARBS.items():
        class_dir = images_dir / food_class
        if not class_dir.exists():
            print(f"  ⚠  Skipping {food_class} (not in Food-101)")
            continue

        images = sorted(class_dir.glob("*.jpg"))[:IMAGES_PER_CLASS]

        for img_path in images:
            key     = f"{food_class}/{img_path.stem}"
            split   = ("train" if hash(img_path.name) % 5 != 0 else "val") \
                      if key in train_keys else "test"

            portion = estimate_portion(img_path)

            # Carb range ground truth
            final_carbs = int(base_carbs * PORTION_MULTIPLIER[portion])
            carb_range  = grams_to_range(final_carbs)
            if split == "test":
                # final_eval - unseen holdout used for evaluation only
                dest3 = eval_dir / f"range_{carb_range}"
                dest3.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dest3 / f"{food_class}_{img_path.name}")
                stats["final"][str(carb_range)] += 1
            else:
                # train_eval - Food-101 TRAIN images labelled by carb range
                dest4 = train_dir / f"range_{carb_range}"
                dest4.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dest4 / f"{food_class}_{img_path.name}")
                stats["train"][str(carb_range)] += 1

            records.append({
                "image":       str(img_path),
                "food_class":  food_class,
                "split":       split,
                "portion":     portion,
                "base_carbs":  base_carbs,
                "final_carbs": final_carbs,
                "carb_range":  carb_range,
            })

    Path(DATASET_DIR).mkdir(parents=True, exist_ok=True)
    with open(Path(DATASET_DIR) / "metadata.json", "w") as f:
        json.dump({"stats": stats, "records": records}, f, indent=2)

    print("\nOK Datasets built.")
    print(f"\n  Train eval (carb range counts): {stats['train']}")
    print(f"  Final eval (carb range counts): {stats['final']}")
