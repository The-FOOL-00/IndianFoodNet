#!/usr/bin/env python3
"""
add_other_class.py
Adds a 4th "other" class to an existing 3-class processed food dataset.
Sources: Indian Food dataset + Food-101 dataset.
"""

import hashlib
import random
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
import albumentations as A

# ─── Config ───────────────────────────────────────────────────────────────────

PROCESSED_DIR   = Path("data/processed")
INDIAN_FOOD_DIR = Path("C:/Users/Sundareswaran/Downloads/food_ds1/Indian Food Images/Indian Food Images")
FOOD101_DIR     = Path("C:/Users/Sundareswaran/Downloads/food-101/food-101/images")

# deduplicate in-place (dal_tadka and biryani appear twice in spec)
INDIAN_FOLDERS: List[str] = list(dict.fromkeys([
    "biryani", "butter_chicken", "dal_makhani", "palak_paneer",
    "chana_masala", "kadai_paneer", "dal_tadka", "lassi", "rabri",
    "shrikhand", "phirni", "poha", "chapati", "naan", "aloo_gobi",
    "aloo_matar", "aloo_methi", "aloo_shimla_mirch", "bhindi_masala",
    "gajar_ka_halwa", "sohan_halwa", "basundi", "doodhpak",
    "double_ka_meetha", "qubani_ka_meetha", "chak_hao_kheer",
    "ras_malai", "misti_doi", "sheera", "sheer_korma", "kalakand",
    "dharwad_pedha", "chicken_tikka_masala", "chicken_razala",
    "navrattan_korma", "paneer_butter_masala", "maach_jhol", "karela_bharta",
]))

FOOD101_FOLDERS: List[str] = [
    "ice_cream", "pizza", "pasta", "caesar_salad", "greek_salad",
    "cheesecake", "tiramisu", "sushi", "ramen", "pho",
    "chocolate_cake", "macarons", "cannoli", "creme_brulee",
    "caprese_salad", "guacamole", "hummus", "nachos", "tacos",
    "lasagna", "ravioli", "risotto", "paella", "bibimbap",
]

TARGET_TOTAL  = 3200
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.10
RANDOM_SEED   = 42
IMG_SIZE      = 224
MIN_SIZE      = 200
CLASS_NAME    = "other"
IMAGE_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─── Augmentation pipelines (5 per image) ─────────────────────────────────────

AUGMENTATIONS: List[Tuple[str, A.Compose]] = [
    ("hflip",       A.Compose([A.HorizontalFlip(p=1.0)])),
    ("bright",      A.Compose([A.RandomBrightnessContrast(brightness_limit=0.3,  contrast_limit=0.0,  p=1.0)])),
    ("contrast",    A.Compose([A.RandomBrightnessContrast(brightness_limit=0.0,  contrast_limit=0.3,  p=1.0)])),
    ("rotate",      A.Compose([A.Rotate(limit=15, p=1.0)])),
    ("flip_bright", A.Compose([A.HorizontalFlip(p=1.0),
                               A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.0, p=1.0)])),
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def md5_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_images(base: Path, folders: List[str]) -> List[Path]:
    images: List[Path] = []
    for folder in folders:
        folder_path = base / folder
        if not folder_path.exists():
            print(f"  [WARN] folder not found: {folder_path}")
            continue
        for p in folder_path.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                images.append(p)
    return images


def validate_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            w, h = img.size
            return w >= MIN_SIZE and h >= MIN_SIZE
    except Exception:
        return False


def load_resize(path: Path) -> Optional[Image.Image]:
    try:
        with Image.open(path) as img:
            return img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    except Exception:
        return None


def save_jpg(img: Image.Image, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, "JPEG", quality=95)


def copy_split(paths: List[Path], dest_dir: Path, label: str) -> int:
    saved = 0
    skipped = 0
    for i, src in enumerate(paths, start=1):
        img = load_resize(src)
        if img is None:
            skipped += 1
            continue
        save_jpg(img, dest_dir / f"other_{i:05d}.jpg")
        saved += 1
        if saved % 200 == 0:
            print(f"    [{label}] {saved}/{len(paths)} copied ...", flush=True)
    if skipped:
        print(f"  [SKIP] {skipped} files failed to load in {label}")
    return saved


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── 1. Collect ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Collecting candidate images ...")
    indian  = collect_images(INDIAN_FOOD_DIR, INDIAN_FOLDERS)
    food101 = collect_images(FOOD101_DIR, FOOD101_FOLDERS)
    all_paths = indian + food101
    print(f"  Indian Food: {len(indian)} images from {len(INDIAN_FOLDERS)} folders")
    print(f"  Food-101:    {len(food101)} images from {len(FOOD101_FOLDERS)} folders")
    print(f"  Raw total:   {len(all_paths)}")

    # ── 2. Filter ─────────────────────────────────────────────────────────────
    print(f"\nStep 2: Filtering corrupted / undersized images (< {MIN_SIZE}px) ...")
    valid: List[Path] = []
    skipped_filter = 0
    for p in all_paths:
        if validate_image(p):
            valid.append(p)
        else:
            skipped_filter += 1
    print(f"  Valid: {len(valid)}  (skipped {skipped_filter})")

    # ── 3. MD5 deduplication ─────────────────────────────────────────────────
    print("\nStep 3: MD5 deduplication ...")
    seen: Set[str] = set()
    unique: List[Path] = []
    for p in valid:
        digest = md5_hash(p)
        if digest not in seen:
            seen.add(digest)
            unique.append(p)
    print(f"  Unique: {len(unique)}  (removed {len(valid) - len(unique)} duplicates)")

    # ── 4. Sample to target ───────────────────────────────────────────────────
    print(f"\nStep 4: Sampling up to {TARGET_TOTAL} images ...")
    if len(unique) < TARGET_TOTAL:
        print(f"  [WARN] Only {len(unique)} unique images available; "
              f"target was {TARGET_TOTAL}. Proceeding with all available.")
        selected = unique[:]
    else:
        random.shuffle(unique)
        selected = unique[:TARGET_TOTAL]
    print(f"  Selected: {len(selected)}")

    # ── 5. Split 70 / 10 / 20 ────────────────────────────────────────────────
    n       = len(selected)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    n_test  = n - n_train - n_val

    train_src = selected[:n_train]
    val_src   = selected[n_train : n_train + n_val]
    test_src  = selected[n_train + n_val :]

    print(f"\nStep 5: Split (seed={RANDOM_SEED})")
    print(f"  train_src={len(train_src)}, val_src={len(val_src)}, test_src={len(test_src)}")

    # ── 6. Create destination dirs ────────────────────────────────────────────
    dest_train = PROCESSED_DIR / "train" / CLASS_NAME
    dest_val   = PROCESSED_DIR / "val"   / CLASS_NAME
    dest_test  = PROCESSED_DIR / "test"  / CLASS_NAME
    for d in (dest_train, dest_val, dest_test):
        d.mkdir(parents=True, exist_ok=True)

    # ── 7. Copy val and test ──────────────────────────────────────────────────
    print("\nStep 6: Copying val images ...")
    val_saved = copy_split(val_src, dest_val, "val")

    print("\nStep 7: Copying test images ...")
    test_saved = copy_split(test_src, dest_test, "test")

    # ── 8. Copy train + augment ───────────────────────────────────────────────
    print("\nStep 8: Copying train images + generating augmentations ...")
    train_orig = 0
    aug_saved  = 0
    skipped_train = 0

    for i, src in enumerate(train_src, start=1):
        img = load_resize(src)
        if img is None:
            skipped_train += 1
            continue

        stem = f"other_{i:05d}"
        save_jpg(img, dest_train / f"{stem}.jpg")
        train_orig += 1

        arr = np.array(img)
        for aug_name, pipeline in AUGMENTATIONS:
            augmented = pipeline(image=arr)["image"]
            save_jpg(Image.fromarray(augmented), dest_train / f"{stem}_{aug_name}.jpg")
            aug_saved += 1

        if train_orig % 200 == 0:
            print(f"  [train] {train_orig}/{len(train_src)} original + {aug_saved} augmented ...",
                  flush=True)

    if skipped_train:
        print(f"  [SKIP] {skipped_train} files failed to load in train")

    # ── 9. Final summary ──────────────────────────────────────────────────────
    train_total = len(list(dest_train.glob("*.jpg")))
    val_total   = len(list(dest_val.glob("*.jpg")))
    test_total  = len(list(dest_test.glob("*.jpg")))

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"train/other: {train_total:>6,} images  "
          f"({train_orig} original + {aug_saved} augmented)")
    print(f"val/other:   {val_total:>6,} images")
    print(f"test/other:  {test_total:>6,} images")
    print(f"Grand total: {train_total + val_total + test_total:>6,} images")
    print("=" * 60)


if __name__ == "__main__":
    main()
