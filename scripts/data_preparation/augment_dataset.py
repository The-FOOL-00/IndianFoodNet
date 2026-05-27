"""
augment_dataset.py
------------------
Augments ONLY the training split of the Indian food dataset.
Val/test splits are intentionally left untouched (research-valid evaluation).

Classes : fried | steamed | grilled
Source  : data/processed/train/{class}/
Output  : same folders, new files named {stem}_aug1.jpg … {stem}_aug5.jpg

Augmentations applied per image
  aug1 – Horizontal flip
  aug2 – Brightness  ×[0.70 – 1.30]
  aug3 – Contrast    ×[0.80 – 1.20]
  aug4 – Rotation    ±15°
  aug5 – Combined: horizontal flip + brightness

Usage
  python augment_dataset.py            # augment + print stats + save figure
"""

import os
import random
import cv2
import numpy as np
import albumentations as A
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ─── Paths & constants ────────────────────────────────────────────────────────

TRAIN_DIR   = Path("data/processed/train")
RESULTS_DIR = Path("results/plots")
CLASSES     = ["fried", "steamed", "grilled"]

# ─── Augmentation pipelines ───────────────────────────────────────────────────
# All pipelines use p=1.0 so the transform is always applied when called.
# Randomness comes from albumentations' internal RNG (seeded per call).

AUGMENTATIONS = {
    "aug1": A.Compose([
        A.HorizontalFlip(p=1.0),
    ]),

    "aug2": A.Compose([
        # brightness factor sampled uniformly from [0.70, 1.30]
        # contrast_limit=(0,0) → no contrast change
        A.RandomBrightnessContrast(
            brightness_limit=(-0.30, 0.30),
            contrast_limit=(0.0, 0.0),
            p=1.0,
        ),
    ]),

    "aug3": A.Compose([
        # contrast factor sampled uniformly from [0.80, 1.20]
        # brightness_limit=(0,0) → no brightness change
        A.RandomBrightnessContrast(
            brightness_limit=(0.0, 0.0),
            contrast_limit=(-0.20, 0.20),
            p=1.0,
        ),
    ]),

    "aug4": A.Compose([
        # angle sampled uniformly from [-15°, +15°]
        # REFLECT_101 avoids black border artefacts
        A.Rotate(
            limit=15,
            border_mode=cv2.BORDER_REFLECT_101,
            p=1.0,
        ),
    ]),

    "aug5": A.Compose([
        # combined: flip  then  random brightness [0.70, 1.30]
        A.HorizontalFlip(p=1.0),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.30, 0.30),
            contrast_limit=(0.0, 0.0),
            p=1.0,
        ),
    ]),
}

AUG_LABELS = {
    "aug1": "Horizontal Flip",
    "aug2": "Brightness",
    "aug3": "Contrast",
    "aug4": "Rotation ±15°",
    "aug5": "Flip + Brightness",
}

# ─── I/O helpers ──────────────────────────────────────────────────────────────

def load_image(path: Path) -> np.ndarray:
    """Read a JPEG as an RGB uint8 numpy array."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_image(img: np.ndarray, path: Path, quality: int = 95) -> None:
    """Write an RGB uint8 numpy array as JPEG."""
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])


def get_originals(class_dir: Path) -> list[Path]:
    """Return original (non-augmented) JPEG paths in a class folder."""
    return sorted(
        p for p in class_dir.glob("*.jpg")
        if "_aug" not in p.stem
    )

# ─── Core augmentation routine ────────────────────────────────────────────────

def augment_training_set() -> dict:
    """
    For every original image in TRAIN_DIR/<class>/ apply each pipeline in
    AUGMENTATIONS and save the result alongside the source file.

    Already-augmented files (stem contains '_aug') are detected and skipped
    so the script is safe to re-run without creating duplicates.

    Returns
    -------
    stats : dict  {class_name: {"original": int, "new": int, "total": int}}
    """
    stats = {}

    for cls in CLASSES:
        class_dir = TRAIN_DIR / cls

        if not class_dir.exists():
            print(f"  [WARN] {class_dir} not found – skipping class '{cls}'.")
            stats[cls] = {"original": 0, "new": 0, "total": 0}
            continue

        originals = get_originals(class_dir)
        n_original = len(originals)

        new_files = 0
        skipped   = 0

        for img_path in tqdm(originals, desc=f"  {cls:<10}", unit="img", ncols=72):
            image = load_image(img_path)
            stem  = img_path.stem          # e.g. "fried_001"

            for aug_tag, pipeline in AUGMENTATIONS.items():
                out_path = class_dir / f"{stem}_{aug_tag}.jpg"

                if out_path.exists():      # safe re-run: skip existing files
                    skipped += 1
                    continue

                augmented = pipeline(image=image)["image"]
                save_image(augmented, out_path)
                new_files += 1

        total = len(list(class_dir.glob("*.jpg")))
        stats[cls] = {
            "original": n_original,
            "new":      new_files,
            "total":    total,
        }

        if skipped:
            print(f"    ↳ {skipped} aug files already existed and were skipped.")

    return stats

# ─── Statistics display ───────────────────────────────────────────────────────

def print_statistics(stats: dict) -> None:
    """Print a formatted table of per-class and total image counts."""
    sep = "─" * 62

    print(f"\n{sep}")
    print(f"  {'CLASS':<12}  {'ORIGINAL':>10}  {'NEW AUGs':>10}  {'TOTAL':>8}")
    print(sep)

    grand_orig  = 0
    grand_new   = 0
    grand_total = 0

    for cls in CLASSES:
        if cls not in stats:
            continue
        c = stats[cls]
        print(
            f"  {cls:<12}  {c['original']:>10}  {c['new']:>10}  {c['total']:>8}"
        )
        grand_orig  += c["original"]
        grand_new   += c["new"]
        grand_total += c["total"]

    print(sep)
    print(
        f"  {'TOTAL':<12}  {grand_orig:>10}  {grand_new:>10}  {grand_total:>8}"
    )
    print(sep)
    print(
        f"\n  Augmentation factor : ×{grand_total / grand_orig:.1f}  "
        f"({len(AUGMENTATIONS)} transforms per image)"
    )
    print("  Val / Test splits   : untouched  (research-valid)")

# ─── Figure for paper ─────────────────────────────────────────────────────────

def show_augmentation_examples(seed: int = 42) -> None:
    """
    Pick one random original image per class, apply every augmentation,
    and save a publication-ready grid figure.

    Layout  : rows = classes (fried / steamed / grilled)
              cols = Original | aug1 | aug2 | aug3 | aug4 | aug5

    Output  : results/plots/augmentation_examples.png  (180 dpi)
    """
    random.seed(seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n_rows = len(CLASSES)
    n_cols = 1 + len(AUGMENTATIONS)     # original + 5 augmented

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols, 3.0 * n_rows),
        gridspec_kw={"wspace": 0.04, "hspace": 0.10},
    )

    col_titles = ["Original"] + [AUG_LABELS[k] for k in AUGMENTATIONS]

    for row, cls in enumerate(CLASSES):
        class_dir = TRAIN_DIR / cls
        originals = get_originals(class_dir) if class_dir.exists() else []

        if not originals:
            print(f"  [WARN] No original images for '{cls}' – row left blank.")
            for ax in axes[row]:
                ax.axis("off")
            continue

        img_path = random.choice(originals)
        image    = load_image(img_path)

        # Column 0 – original
        axes[row, 0].imshow(image)
        axes[row, 0].set_ylabel(
            cls.capitalize(),
            fontsize=13,
            fontweight="bold",
            rotation=90,
            labelpad=8,
            va="center",
        )

        # Columns 1-5 – augmented
        for col_offset, (aug_tag, pipeline) in enumerate(AUGMENTATIONS.items(), start=1):
            augmented = pipeline(image=image)["image"]
            axes[row, col_offset].imshow(augmented)

        # Column titles on first row only
        if row == 0:
            for col, title in enumerate(col_titles):
                axes[row, col].set_title(title, fontsize=10, pad=5)

        # Remove all tick marks
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "Data Augmentation Examples – Indian Food Dataset\n"
        "Training split only  |  val/test untouched",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    out_path = RESULTS_DIR / "augmentation_examples.png"
    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    print(f"\n  Figure saved → {out_path.resolve()}")

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  Indian Food Dataset  –  Training-Set Augmentation")
    print(f"  Source : {TRAIN_DIR.resolve()}")
    print(f"  Classes: {', '.join(CLASSES)}")
    print("=" * 62)

    print("\n[1/3]  Augmenting training images …")
    stats = augment_training_set()

    print("\n[2/3]  Dataset statistics:")
    print_statistics(stats)

    print("\n[3/3]  Generating augmentation examples figure …")
    show_augmentation_examples()

    print("\nAll done.\n")
