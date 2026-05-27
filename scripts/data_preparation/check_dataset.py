#!/usr/bin/env python3
"""
check_dataset.py
Validate and summarise the Indian food cooking-method image dataset.

Dependencies:
    pip install pillow matplotlib
"""

import sys
from pathlib import Path
from io import BytesIO

from PIL import Image, UnidentifiedImageError
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on all platforms)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ──────────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("data/processed/train")
PLOT_DIR  = Path("results/plots")
CLASSES   = ["fried", "steamed", "grilled", "other"]
IMAGE_EXT = {".jpg", ".jpeg", ".png"}

# ──────────────────────────────────────────────────────────────────────────────
#  Audit helpers
# ──────────────────────────────────────────────────────────────────────────────

def audit_class(cls: str) -> dict:
    """
    Walk one class folder and return statistics + corruption list.
    """
    cls_dir = DATA_DIR / cls
    widths: list[int]    = []
    heights: list[int]   = []
    corrupted: list[str] = []
    sizes_bytes: list[int] = []

    if not cls_dir.exists():
        return {
            "count": 0, "valid": 0,
            "widths": [], "heights": [],
            "sizes_bytes": [], "corrupted": [],
        }

    candidates = [p for p in sorted(cls_dir.iterdir()) if p.suffix.lower() in IMAGE_EXT]

    for img_path in candidates:
        try:
            with Image.open(img_path) as img:
                img.verify()                    # raises on truncated / invalid files
            with Image.open(img_path) as img:   # must re-open after verify
                w, h = img.size
            widths.append(w)
            heights.append(h)
            sizes_bytes.append(img_path.stat().st_size)
        except (UnidentifiedImageError, OSError, Exception):
            corrupted.append(img_path.name)

    return {
        "count":       len(candidates),
        "valid":       len(widths),
        "widths":      widths,
        "heights":     heights,
        "sizes_bytes": sizes_bytes,
        "corrupted":   corrupted,
    }


def _stats(values: list[int | float]) -> dict:
    if not values:
        return {"min": 0, "max": 0, "avg": 0.0, "median": 0.0}
    s = sorted(values)
    n = len(s)
    mid = n // 2
    median = s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2
    return {
        "min":    s[0],
        "max":    s[-1],
        "avg":    sum(s) / n,
        "median": median,
    }


def _human_bytes(b: float) -> str:
    for unit in ("B", "KB", "MB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} GB"


# ──────────────────────────────────────────────────────────────────────────────
#  Report printer
# ──────────────────────────────────────────────────────────────────────────────

def print_report(results: dict[str, dict]) -> None:
    sep  = "=" * 64
    sep2 = "-" * 64

    print(f"\n{sep}")
    print("  Indian Food Dataset – Quality Report")
    print(sep)

    total_valid = 0
    all_corrupted: list[tuple[str, str]] = []

    for cls in CLASSES:
        r  = results[cls]
        ws = _stats(r["widths"])
        hs = _stats(r["heights"])
        fs = _stats(r["sizes_bytes"])

        total_valid += r["valid"]
        for name in r["corrupted"]:
            all_corrupted.append((cls, name))

        print(f"\n  CLASS: {cls.upper()}")
        print(f"    Files found    : {r['count']}")
        print(f"    Valid images   : {r['valid']}")
        print(f"    Corrupted      : {len(r['corrupted'])}")

        if r["widths"]:
            print(f"    Width  (px)    : "
                  f"min={ws['min']}  max={ws['max']}  "
                  f"avg={ws['avg']:.0f}  median={ws['median']:.0f}")
            print(f"    Height (px)    : "
                  f"min={hs['min']}  max={hs['max']}  "
                  f"avg={hs['avg']:.0f}  median={hs['median']:.0f}")
            print(f"    File size      : "
                  f"min={_human_bytes(fs['min'])}  "
                  f"max={_human_bytes(fs['max'])}  "
                  f"avg={_human_bytes(fs['avg'])}")
        else:
            print("    No valid images found.")

        if r["corrupted"]:
            print("    Corrupted files:")
            for name in r["corrupted"]:
                print(f"      ✗  {name}")

    print(f"\n{sep2}")
    print(f"  Total valid images : {total_valid}")
    if all_corrupted:
        print(f"  Total corrupted    : {len(all_corrupted)}")
        print("  (see per-class listing above)")
    print(sep)


# ──────────────────────────────────────────────────────────────────────────────
#  Bar chart
# ──────────────────────────────────────────────────────────────────────────────

_CLASS_COLORS = {
    "fried":   "#E07B54",   # warm orange
    "steamed": "#5BA4CF",   # cool blue
    "grilled": "#6ABF69",   # green
    "other":   "#A98FD4",   # muted purple
}


def generate_chart(results: dict[str, dict]) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    counts = [results[cls]["valid"] for cls in CLASSES]
    colors = [_CLASS_COLORS[cls] for cls in CLASSES]
    labels = [cls.capitalize() for cls in CLASSES]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(labels, counts, color=colors, width=0.55,
                  edgecolor="white", linewidth=1.5, zorder=3)

    # Value labels on bars
    ax.bar_label(bars, fmt="%d", padding=5, fontsize=12, fontweight="bold", color="#333333")

    # Reference line at target (MAX_PER_CLASS = 400)
    target = 400
    ax.axhline(target, color="#999999", linewidth=1.2, linestyle="--", zorder=2)
    ax.text(len(CLASSES) - 0.5, target + 5, f"target ({target})",
            color="#888888", fontsize=9, ha="right")

    ax.set_title("Indian Food Dataset – Class Distribution",
                 fontsize=14, fontweight="bold", pad=16)
    ax.set_xlabel("Cooking Method", fontsize=12, labelpad=8)
    ax.set_ylabel("Number of Images", fontsize=12, labelpad=8)

    max_count = max(counts + [target])
    ax.set_ylim(0, max_count * 1.18)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Percentage labels below bar labels
    total = sum(counts) or 1
    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{pct:.1f}%",
            ha="center", va="center",
            fontsize=10, color="white", fontweight="bold",
            alpha=0.85,
        )

    plt.tight_layout()
    out_path = PLOT_DIR / "class_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
#  Sanity checks
# ──────────────────────────────────────────────────────────────────────────────

def check_balance(results: dict[str, dict]) -> None:
    counts = [results[cls]["valid"] for cls in CLASSES]
    if not any(counts):
        return
    mx = max(counts)
    mn = min(c for c in counts if c > 0)
    ratio = mx / mn if mn else float("inf")
    print(f"\n  Class balance ratio (max/min): {ratio:.2f}x")
    if ratio > 2:
        print("  WARNING: dataset is imbalanced (ratio > 2). Consider collecting more data")
        print("           for the smaller class(es) before training.")
    else:
        print("  Class balance looks acceptable (ratio ≤ 2).")


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not DATA_DIR.exists():
        print(f"ERROR: data directory not found: {DATA_DIR.resolve()}")
        print("Run scrape_images.py first.")
        sys.exit(1)

    print("Auditing dataset…")
    results = {cls: audit_class(cls) for cls in CLASSES}

    print_report(results)
    check_balance(results)

    out_path = generate_chart(results)
    print(f"\n  Chart saved → {out_path.resolve()}")
    print()


if __name__ == "__main__":
    main()
