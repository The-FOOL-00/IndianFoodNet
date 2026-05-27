#!/usr/bin/env python3
"""build_final_dataset.py — Build clean Indian food cooking-method classification dataset."""

import hashlib
import random
import shutil
from pathlib import Path

from PIL import Image

# ── Source roots ──────────────────────────────────────────────────────────────
INDIAN_ROOT  = Path("C:/Users/Sundareswaran/Downloads/food_ds1/Indian Food Images/Indian Food Images")
FOOD101_ROOT = Path("C:/Users/Sundareswaran/Downloads/food-101/food-101/images")

# ── Destination ───────────────────────────────────────────────────────────────
DST_ROOT  = Path("data/raw")
DST = {
    "fried":   DST_ROOT / "fried",
    "steamed": DST_ROOT / "steamed",
    "grilled": DST_ROOT / "grilled",
}
DST_OTHER = DST_ROOT / "other"

# ── Per-class source folder names ─────────────────────────────────────────────
SOURCES = {
    "fried": {
        "indian": [
            "bhatura", "jalebi", "kachori", "aloo_tikki", "anarsa",
            "gavvalu", "ghevar", "imarti", "shankarpali", "gulab_jamun",
            "kajjikaya", "ariselu", "adhirasam", "kakinada_khaja",
            "boondi", "sutar_feni", "mysore_pak",
        ],
        "food101": [
            "samosa", "spring_rolls", "onion_rings", "french_fries",
            "donuts", "churros", "fried_calamari", "beignets", "falafel",
            "takoyaki",
        ],
    },
    "steamed": {
        "indian": [
            "idli", "dhokla", "modak", "unni_appam", "kuzhi_paniyaram",
            "pithe", "rasgulla", "sandesh", "cham_cham",
            "poornalu", "ledikeni", "lyangcha", "chhena_kheeri",
        ],
        "food101": [
            "dumplings", "edamame", "gyoza",
        ],
    },
    "grilled": {
        "indian": [
            "chicken_tikka", "litti_chokha",
        ],
        "food101": [
            "baby_back_ribs", "grilled_salmon", "steak",
            "filet_mignon", "pork_chop", "prime_rib",
        ],
    },
}

VALID_EXTS = {".jpg", ".jpeg", ".png"}
MIN_DIM    = 224
SEED       = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_image(path: Path) -> str:
    """Return 'ok', 'small', or 'corrupt'."""
    try:
        with Image.open(path) as img:
            w, h = img.size
            if w < MIN_DIM or h < MIN_DIM:
                return "small"
            img.load()  # full decode — raises on truncated/broken data
    except Exception:
        return "corrupt"
    return "ok"


def find_source_dir(root: Path, name: str) -> Path | None:
    """Return matching subdirectory under root (case-insensitive), or None."""
    direct = root / name
    if direct.exists() and direct.is_dir():
        return direct
    if root.exists():
        lower = name.lower()
        for d in root.iterdir():
            if d.is_dir() and d.name.lower() == lower:
                return d
    return None


def copy_class_images(
    root: Path,
    folder_names: list[str],
    dst_dir: Path,
    seen_hashes: set[str],
    stats: dict,
) -> None:
    for name in folder_names:
        src_dir = find_source_dir(root, name)
        if src_dir is None:
            print(f"    [WARN] Not found: {root / name}")
            stats["missing"] += 1
            continue

        for img_path in sorted(src_dir.iterdir()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in VALID_EXTS:
                continue

            status = validate_image(img_path)
            if status == "small":
                stats["small"] += 1
                continue
            if status == "corrupt":
                stats["corrupt"] += 1
                continue

            digest = md5_of_file(img_path)
            if digest in seen_hashes:
                stats["duplicate"] += 1
                continue
            seen_hashes.add(digest)

            # Avoid name collision within destination folder
            dst = dst_dir / img_path.name
            if dst.exists():
                stem   = img_path.stem
                suffix = img_path.suffix.lower()
                n = 1
                while dst.exists():
                    dst = dst_dir / f"{stem}_{n}{suffix}"
                    n += 1

            shutil.copy2(img_path, dst)


def two_pass_rename(dst_dir: Path, cls: str) -> None:
    """Rename all images to cls_001.ext … without collisions."""
    files = sorted(
        f for f in dst_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTS
    )

    # Pass 1 — atomic temp names to avoid any source/target overlap
    tmp_paths: list[Path] = []
    for i, f in enumerate(files):
        tmp = dst_dir / f"__tmp_{i:08d}{f.suffix.lower()}"
        f.rename(tmp)
        tmp_paths.append(tmp)

    # Pass 2 — final clean names (sort for deterministic ordering)
    for i, tmp in enumerate(sorted(tmp_paths), start=1):
        final = dst_dir / f"{cls}_{i:03d}{tmp.suffix}"
        tmp.rename(final)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(SEED)
    stats = {
        "missing":  0,
        "small":    0,
        "corrupt":  0,
        "duplicate": 0,
        "balanced": 0,
    }

    # ── STEP 1: Clear destination folders ─────────────────────────────────────
    print("STEP 1: Clearing destination folders...")
    for dst_dir in DST.values():
        dst_dir.mkdir(parents=True, exist_ok=True)
        for item in dst_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        print(f"  Cleared: {dst_dir}")

    if DST_OTHER.exists():
        shutil.rmtree(DST_OTHER)
        print(f"  Deleted: {DST_OTHER}")

    # ── STEP 2 + 3: Copy with quality filters ─────────────────────────────────
    print("\nSTEP 2+3: Copying and filtering images...")
    seen_hashes: set[str] = set()  # shared across all classes for cross-class dedup

    for cls, dst_dir in DST.items():
        print(f"\n  [{cls}]")
        copy_class_images(INDIAN_ROOT,  SOURCES[cls]["indian"],  dst_dir, seen_hashes, stats)
        copy_class_images(FOOD101_ROOT, SOURCES[cls]["food101"], dst_dir, seen_hashes, stats)
        count = sum(1 for f in dst_dir.iterdir() if f.is_file())
        print(f"  -> {count} images collected")

    # ── STEP 4: Balance classes ────────────────────────────────────────────────
    print("\nSTEP 4: Balancing classes...")
    class_files: dict[str, list[Path]] = {}
    for cls, dst_dir in DST.items():
        class_files[cls] = [
            f for f in dst_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_EXTS
        ]
        print(f"  {cls}: {len(class_files[cls])} images before balancing")

    min_count = min(len(v) for v in class_files.values())
    print(f"  Target (min class): {min_count}")

    for cls, files in class_files.items():
        excess = len(files) - min_count
        if excess > 0:
            to_remove = random.sample(files, excess)
            for f in to_remove:
                f.unlink()
            stats["balanced"] += excess
            print(f"  {cls}: removed {excess} images")

    # ── STEP 5: Rename cleanly ─────────────────────────────────────────────────
    print("\nSTEP 5: Renaming files (two-pass)...")
    for cls, dst_dir in DST.items():
        two_pass_rename(dst_dir, cls)
        print(f"  {cls}: done")

    # ── STEP 6: Final report ───────────────────────────────────────────────────
    final_counts = {
        cls: sum(
            1 for f in dst_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_EXTS
        )
        for cls, dst_dir in DST.items()
    }
    total  = sum(final_counts.values())
    counts = list(final_counts.values())
    ratio  = max(counts) / min(counts) if min(counts) > 0 else float("inf")

    print("\n" + "=" * 48)
    print("FINAL DATASET SUMMARY")
    print("=" * 48)
    print(f"fried:   {final_counts['fried']:>5} images")
    print(f"steamed: {final_counts['steamed']:>5} images")
    print(f"grilled: {final_counts['grilled']:>5} images")
    print(f"total:   {total:>5} images")
    print(f"class balance ratio: {ratio:.2f}")
    print("=" * 48)
    print(f"Source folders not found (warnings): {stats['missing']}")
    print(f"Images rejected (too small):         {stats['small']}")
    print(f"Images rejected (corrupted):         {stats['corrupt']}")
    print(f"Images rejected (duplicate):         {stats['duplicate']}")
    print(f"Images removed for balancing:        {stats['balanced']}")
    print("=" * 48)


if __name__ == "__main__":
    main()
