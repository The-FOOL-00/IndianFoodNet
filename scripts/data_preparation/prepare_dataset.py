from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# ── Configuration ─────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
CLASSES = ["fried", "steamed", "grilled"]
SPLIT_RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}
SPLIT_ORDER = ["train", "val", "test"]
IMAGE_SIZE = (224, 224)
SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ── Placeholder detection ─────────────────────────────────────────────────────

def is_placeholder(path: Path) -> tuple[bool, str | None]:
    try:
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        total = arr.shape[0] * arr.shape[1]
        if total == 0:
            return True, "empty_image"

        # More than 90% white pixels (all channels > 240)
        if np.all(arr > 240, axis=2).sum() / total > 0.90:
            return True, "mostly_white"

        # More than 90% black pixels (all channels < 15)
        if np.all(arr < 15, axis=2).sum() / total > 0.90:
            return True, "mostly_black"

        # More than 90% single dominant color channel
        # A pixel is "dominated" by channel c when c is strictly highest
        # and at least 30 above the next channel (avoids flagging gray images).
        for ch, color in enumerate(["red", "green", "blue"]):
            ch_vals = arr[:, :, ch]
            others = np.stack([arr[:, :, i] for i in range(3) if i != ch], axis=-1)
            other_max = others.max(axis=-1)
            dominated = (ch_vals > other_max) & ((ch_vals - other_max) > 30)
            if dominated.sum() / total > 0.90:
                return True, f"single_color_{color}"

        return False, None
    except Exception:
        return True, "unreadable"


# ── Helpers ───────────────────────────────────────────────────────────────────

def collect_images(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def stratified_split(
    images_by_class: dict[str, list[Path]],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, dict[str, list[Path]]]:
    rng = random.Random(seed)
    result: dict[str, dict[str, list[Path]]] = {s: {} for s in ratios}
    for cls, images in images_by_class.items():
        shuffled = list(images)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = round(n * ratios["train"])
        n_val = round(n * ratios["val"])
        result["train"][cls] = shuffled[:n_train]
        result["val"][cls] = shuffled[n_train : n_train + n_val]
        result["test"][cls] = shuffled[n_train + n_val :]
    return result


def copy_resized(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGB")
    img = img.resize(IMAGE_SIZE, Image.LANCZOS)
    img.save(dst)


def fmt_row(label: str, value: int, col: int = 11) -> str:
    prefix = f"  {label}:"
    padding = max(1, col - len(prefix))
    return f"{prefix}{' ' * padding} {value}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── STEP 1: Scan and remove placeholder images ────────────────────────────
    print("STEP 1: Scanning for placeholder images...")
    removed_log: list[dict[str, str]] = []
    surviving: dict[str, list[Path]] = {}

    for cls in CLASSES:
        keep: list[Path] = []
        for path in collect_images(RAW_DIR / cls):
            flagged, reason = is_placeholder(path)
            if flagged:
                removed_log.append({"filename": path.name, "class": cls, "reason": reason})
                path.unlink()
            else:
                keep.append(path)
        surviving[cls] = keep
        n_rm = sum(1 for r in removed_log if r["class"] == cls)
        print(f"  {cls}: {len(keep)} kept, {n_rm} removed")

    csv_path = RESULTS_DIR / "placeholder_removed.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "class", "reason"])
        writer.writeheader()
        writer.writerows(removed_log)
    print(f"  Log saved: {csv_path}\n")

    # ── STEP 2 & 3: Stratified split + resize to 224×224 ─────────────────────
    print("STEP 2-3: Splitting and resizing images...")
    splits = stratified_split(surviving, SPLIT_RATIOS, SEED)
    counts: dict[str, dict[str, int]] = {s: {cls: 0 for cls in CLASSES} for s in SPLIT_ORDER}

    for split_name in SPLIT_ORDER:
        for cls in CLASSES:
            for src in splits[split_name][cls]:
                copy_resized(src, PROCESSED_DIR / split_name / cls / src.name)
                counts[split_name][cls] += 1
        print(f"  {split_name}: {sum(counts[split_name].values())} images")

    # ── STEP 4: Build and print report ───────────────────────────────────────
    sep = "=" * 48
    lines = [
        sep,
        "DATASET PREPARATION COMPLETE",
        sep,
        f"Images removed (placeholders): {len(removed_log)}",
        "Remaining images per class:",
    ]
    for cls in CLASSES:
        lines.append(fmt_row(cls, len(surviving[cls])))
    lines.append("")

    for split_name in SPLIT_ORDER:
        split_total = sum(counts[split_name].values())
        lines.append(f"{split_name.upper()} SET:")
        for cls in CLASSES:
            lines.append(fmt_row(cls, counts[split_name][cls]))
        lines.append(fmt_row("total", split_total))
        lines.append("")

    grand_total = sum(sum(counts[s].values()) for s in SPLIT_ORDER)
    lines.append(f"GRAND TOTAL: {grand_total} images")
    lines.append(sep)

    report = "\n".join(lines)
    print("\n" + report)

    report_path = RESULTS_DIR / "dataset_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
