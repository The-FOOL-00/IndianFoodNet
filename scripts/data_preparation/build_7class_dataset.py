#!/usr/bin/env python3
"""
build_7class_dataset.py

Builds a complete 7-class Indian food cooking method classification dataset
from scratch using the Indian Food Images dataset and Food-101 as sources.

Classes: fried, steamed, grilled, baked, boiled, raw, other
"""

import hashlib
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import numpy as np
from PIL import Image

# ── Source roots ───────────────────────────────────────────────────────────────
INDIAN_ROOT = Path(
    "C:/Users/Sundareswaran/Downloads/food_ds1/Indian Food Images/Indian Food Images"
)
FOOD101_ROOT = Path(
    "C:/Users/Sundareswaran/Downloads/food-101/food-101/images"
)
BAKED_SCRAPED = Path("data/raw/baked_scraped")
DEST_ROOT = Path("data/processed")

RANDOM_SEED = 42
MIN_ACCEPTABLE = 500
TARGET_SIZE = (224, 224)
MIN_DIM = 200
VALID_EXTS = {".jpg", ".jpeg", ".png"}
CLASSES = ["fried", "steamed", "grilled", "baked", "boiled", "raw", "other"]

# ── Class → source folder mappings ────────────────────────────────────────────
# Each tuple: (source_type, folder_name)
# source_type: "indian" | "food101" | "scraped"
CLASS_SOURCES: Dict[str, List[Tuple[str, str]]] = {
    "fried": [
        ("indian",  "bhatura"),
        ("indian",  "jalebi"),
        ("indian",  "kachori"),
        ("indian",  "aloo_tikki"),
        ("indian",  "anarsa"),
        ("indian",  "gavvalu"),
        ("indian",  "ghevar"),
        ("indian",  "imarti"),
        ("indian",  "shankarpali"),
        ("indian",  "gulab_jamun"),
        ("indian",  "kajjikaya"),
        ("indian",  "ariselu"),
        ("indian",  "adhirasam"),
        ("indian",  "kakinada_khaja"),
        ("indian",  "boondi"),
        ("indian",  "sutar_feni"),
        ("indian",  "mysore_pak"),
        ("indian",  "daal_puri"),
        ("food101", "samosa"),
        ("food101", "spring_rolls"),
        ("food101", "onion_rings"),
        ("food101", "french_fries"),
        ("food101", "donuts"),
        ("food101", "churros"),
        ("food101", "fried_calamari"),
        ("food101", "beignets"),
        ("food101", "falafel"),
        ("food101", "takoyaki"),
        ("food101", "waffles"),
        ("food101", "french_toast"),
        ("food101", "crab_cakes"),
        ("food101", "eggs_benedict"),
        ("food101", "fish_and_chips"),
        ("food101", "cannoli"),
    ],
    "steamed": [
        ("indian",  "modak"),
        ("indian",  "sandesh"),
        ("indian",  "unni_appam"),
        ("indian",  "kuzhi_paniyaram"),
        ("indian",  "pithe"),
        ("indian",  "rasgulla"),
        ("indian",  "cham_cham"),
        ("indian",  "ledikeni"),
        ("indian",  "lyangcha"),
        ("indian",  "chhena_kheeri"),
        ("food101", "dumplings"),
        ("food101", "edamame"),
        ("food101", "gyoza"),
        ("food101", "mussels"),
        ("food101", "lobster_bisque"),
    ],
    "grilled": [
        ("indian",  "chicken_tikka"),
        ("indian",  "litti_chokha"),
        ("food101", "baby_back_ribs"),
        ("food101", "grilled_salmon"),
        ("food101", "steak"),
        ("food101", "filet_mignon"),
        ("food101", "pork_chop"),
        ("food101", "prime_rib"),
        ("food101", "peking_duck"),
        ("food101", "chicken_wings"),
        ("food101", "hamburger"),
        ("food101", "hot_dog"),
        ("food101", "grilled_cheese_sandwich"),
    ],
    "baked": [
        ("indian",  "naan"),
        ("indian",  "chapati"),
        ("indian",  "misi_roti"),
        ("indian",  "makki_di_roti_sarson_da_saag"),
        ("indian",  "daal_baati_churma"),
        ("food101", "pizza"),
        ("food101", "garlic_bread"),
        ("food101", "bread_pudding"),
        ("food101", "bruschetta"),
        ("scraped", "baked_scraped"),   # skipped if folder absent
    ],
    "boiled": [
        ("indian",  "biryani"),
        ("indian",  "butter_chicken"),
        ("indian",  "dal_makhani"),
        ("indian",  "palak_paneer"),
        ("indian",  "chana_masala"),
        ("indian",  "kadai_paneer"),
        ("indian",  "dal_tadka"),
        ("indian",  "aloo_gobi"),
        ("indian",  "aloo_matar"),
        ("indian",  "aloo_methi"),
        ("indian",  "aloo_shimla_mirch"),
        ("indian",  "bhindi_masala"),
        ("indian",  "chicken_tikka_masala"),
        ("indian",  "chicken_razala"),
        ("indian",  "navrattan_korma"),
        ("indian",  "paneer_butter_masala"),
        ("indian",  "maach_jhol"),
        ("indian",  "karela_bharta"),
        ("indian",  "dum_aloo"),
        ("indian",  "kofta"),
        ("indian",  "poha"),
        ("indian",  "sheer_korma"),
        ("food101", "ramen"),
        ("food101", "risotto"),
        ("food101", "paella"),
        ("food101", "lasagna"),
        ("food101", "ravioli"),
        ("food101", "clam_chowder"),
        ("food101", "french_onion_soup"),
        ("food101", "miso_soup"),
        ("food101", "hot_and_sour_soup"),
        ("food101", "macaroni_and_cheese"),
        ("food101", "spaghetti_bolognese"),
        ("food101", "spaghetti_carbonara"),
        ("food101", "chicken_curry"),
        ("food101", "pad_thai"),
        ("food101", "gnocchi"),
        ("food101", "poutine"),
        ("food101", "shrimp_and_grits"),
    ],
    "raw": [
        ("indian",  "lassi"),
        ("indian",  "shrikhand"),
        ("indian",  "misti_doi"),
        ("indian",  "rabri"),
        ("indian",  "phirni"),
        ("food101", "caesar_salad"),
        ("food101", "greek_salad"),
        ("food101", "caprese_salad"),
        ("food101", "guacamole"),
        ("food101", "hummus"),
        ("food101", "sushi"),
        ("food101", "seaweed_salad"),
        ("food101", "beef_tartare"),
        ("food101", "ceviche"),
        ("food101", "tuna_tartare"),
        ("food101", "oysters"),
        ("food101", "cheese_plate"),
        ("food101", "beet_salad"),
    ],
    "other": [
        ("indian",  "gajar_ka_halwa"),
        ("indian",  "sohan_halwa"),
        ("indian",  "basundi"),
        ("indian",  "doodhpak"),
        ("indian",  "double_ka_meetha"),
        ("indian",  "qubani_ka_meetha"),
        ("indian",  "chak_hao_kheer"),
        ("indian",  "ras_malai"),
        ("indian",  "sheera"),
        ("indian",  "kalakand"),
        ("indian",  "dharwad_pedha"),
        ("indian",  "sohan_papdi"),
        ("indian",  "chikki"),
        ("indian",  "pootharekulu"),
        ("indian",  "bandar_laddu"),
        ("food101", "ice_cream"),
        ("food101", "cheesecake"),
        ("food101", "tiramisu"),
        ("food101", "chocolate_cake"),
        ("food101", "macarons"),
        ("food101", "creme_brulee"),
        ("food101", "strawberry_shortcake"),
        ("food101", "cup_cakes"),
        ("food101", "red_velvet_cake"),
        ("food101", "frozen_yogurt"),
        ("food101", "panna_cotta"),
        ("food101", "apple_pie"),
        ("food101", "baklava"),
        ("food101", "carrot_cake"),
        ("food101", "chocolate_mousse"),
        ("food101", "nachos"),
        ("food101", "tacos"),
        ("food101", "breakfast_burrito"),
        ("food101", "club_sandwich"),
        ("food101", "lobster_roll_sandwich"),
        ("food101", "deviled_eggs"),
        ("food101", "escargots"),
        ("food101", "foie_gras"),
        ("food101", "huevos_rancheros"),
        ("food101", "croque_madame"),
        ("food101", "pancakes"),
        ("food101", "waffles"),   # deduped out if already claimed by fried
        ("food101", "scallops"),
        ("food101", "bibimbap"),
        ("food101", "omelette"),
        ("food101", "oysters"),   # deduped out if already claimed by raw
    ],
}

AUG_NAMES = ["hflip", "bright", "contrast", "rotate", "flip_bright"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _source_root(source_type: str, folder: str) -> Path:
    if source_type == "scraped":
        return BAKED_SCRAPED
    if source_type == "indian":
        return INDIAN_ROOT / folder
    return FOOD101_ROOT / folder


def _is_valid(path: Path) -> bool:
    """Return True when PIL can open the file and both dimensions are ≥ MIN_DIM."""
    if path.suffix.lower() not in VALID_EXTS:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            w, h = img.size
        return w >= MIN_DIM and h >= MIN_DIM
    except Exception:
        return False


def collect_folder(source_type: str, folder: str) -> List[Path]:
    """Return valid image paths from one source folder."""
    root = _source_root(source_type, folder)
    if not root.exists():
        print(f"    [SKIP] not found: {root}")
        return []
    paths = [p for p in root.rglob("*") if p.is_file() and _is_valid(p)]
    return paths


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def open_resized(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)


def save_jpg(img: Image.Image, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, format="JPEG", quality=95)


def build_augmentations() -> list:
    """Return the five augmentation transforms listed in the spec."""
    return [
        # 1. horizontal flip
        A.HorizontalFlip(p=1.0),
        # 2. brightness  (factor 0.7 – 1.3  →  limit ±0.3)
        A.RandomBrightnessContrast(
            brightness_limit=(-0.3, 0.3),
            contrast_limit=(0.0, 0.0),
            p=1.0,
        ),
        # 3. contrast  (factor 0.8 – 1.2  →  limit ±0.2)
        A.RandomBrightnessContrast(
            brightness_limit=(0.0, 0.0),
            contrast_limit=(-0.2, 0.2),
            p=1.0,
        ),
        # 4. rotation  −15° to +15°
        A.Rotate(limit=15, p=1.0),
        # 5. combined flip + brightness
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.3),
                contrast_limit=(0.0, 0.0),
                p=1.0,
            ),
        ]),
    ]


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── STEP 1: Wipe and recreate data/processed/ ──────────────────────────────
    print("=" * 68)
    print("STEP 1  Clearing data/processed/")
    if DEST_ROOT.exists():
        shutil.rmtree(DEST_ROOT)
    DEST_ROOT.mkdir(parents=True)
    print(f"        Cleared → {DEST_ROOT.resolve()}")

    # ── STEP 2: Collect raw images per class ──────────────────────────────────
    print("\nSTEP 2  Collecting source images")
    raw: Dict[str, List[Path]] = {}
    for cls in CLASSES:
        print(f"\n  [{cls.upper()}]")
        pool: List[Path] = []
        for src_type, folder in CLASS_SOURCES[cls]:
            found = collect_folder(src_type, folder)
            if found:
                label = f"{src_type}/{folder}"
                print(f"    {label:<45} {len(found):>5}")
            pool.extend(found)
        raw[cls] = pool
        print(f"  → {cls}: {len(pool)} images")

    # ── STEP 3: Quality filter (applied inline during collection) ─────────────
    print("\nSTEP 3  Quality filter applied during collection")
    print("        Criteria: jpg/jpeg/png, ≥200×200 px, PIL-openable")

    # ── STEP 4: Global MD5 deduplication ──────────────────────────────────────
    print("\nSTEP 4  Global MD5 deduplication")
    seen_hashes: Dict[str, str] = {}   # md5 → first class that claimed it
    deduped: Dict[str, List[Path]] = {cls: [] for cls in CLASSES}
    total_dupes = 0
    for cls in CLASSES:
        for p in raw[cls]:
            h = file_md5(p)
            if h not in seen_hashes:
                seen_hashes[h] = cls
                deduped[cls].append(p)
            else:
                total_dupes += 1
    print(f"        Removed {total_dupes} cross-class duplicates")
    for cls in CLASSES:
        print(f"        {cls:<12}: {len(raw[cls]):>5} → {len(deduped[cls]):>5}")

    # ── STEP 5: Balance all classes to the minimum count ──────────────────────
    print("\nSTEP 5  Balancing classes")
    counts = {cls: len(deduped[cls]) for cls in CLASSES}
    cap = min(counts.values())
    print(f"        Per-class counts : {counts}")
    print(f"        Cap (minimum)    : {cap}")
    if cap < MIN_ACCEPTABLE:
        print(
            f"        WARNING: cap ({cap}) is below minimum acceptable "
            f"({MIN_ACCEPTABLE}) — continuing anyway"
        )

    balanced: Dict[str, List[Path]] = {}
    for cls in CLASSES:
        if counts[cls] < MIN_ACCEPTABLE:
            print(f"        WARNING: {cls} has only {counts[cls]} images (<{MIN_ACCEPTABLE})")
        pool = deduped[cls].copy()
        random.shuffle(pool)
        balanced[cls] = pool[:cap]

    # ── STEP 6: 70 / 10 / 20 split ───────────────────────────────────────────
    print("\nSTEP 6  Splitting 70 / 10 / 20  (seed=42)")
    train_list: Dict[str, List[Path]] = {}
    val_list:   Dict[str, List[Path]] = {}
    test_list:  Dict[str, List[Path]] = {}
    for cls in CLASSES:
        pool = balanced[cls]
        n = len(pool)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.10)
        train_list[cls] = pool[:n_train]
        val_list[cls]   = pool[n_train:n_train + n_val]
        test_list[cls]  = pool[n_train + n_val:]
        print(
            f"        {cls:<12}: n={n}  "
            f"train={len(train_list[cls])}  "
            f"val={len(val_list[cls])}  "
            f"test={len(test_list[cls])}"
        )

    # ── STEP 7: Write val and test (resize only) ──────────────────────────────
    print("\nSTEP 7  Writing val + test (resize to 224×224)")
    for cls in CLASSES:
        for split, img_paths in [("val", val_list[cls]), ("test", test_list[cls])]:
            dest_dir = DEST_ROOT / split / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            ok = 0
            for idx, src in enumerate(img_paths):
                try:
                    img = open_resized(src)
                    save_jpg(img, dest_dir / f"{cls}_{split}_{idx:05d}.jpg")
                    ok += 1
                except Exception as exc:
                    print(f"    [WARN] {split}/{cls}/{src.name}: {exc}")
            print(f"        {cls:<12}/{split:<5}: {ok} images written")

    # ── STEP 8: Write train (resize + 5 augmented copies) ────────────────────
    print("\nSTEP 8  Writing train (resize 224×224 + 5 augmentations per image)")
    augs = build_augmentations()

    train_orig_count:  Dict[str, int] = {}
    train_total_count: Dict[str, int] = {}

    for cls in CLASSES:
        dest_dir = DEST_ROOT / "train" / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        n_orig = 0
        n_aug  = 0

        for idx, src in enumerate(train_list[cls]):
            try:
                img = open_resized(src)
                save_jpg(img, dest_dir / f"{cls}_train_{idx:05d}.jpg")
                n_orig += 1

                img_np = np.array(img)
                for aug_fn, aname in zip(augs, AUG_NAMES):
                    try:
                        aug_arr = aug_fn(image=img_np)["image"]
                        aug_img = Image.fromarray(aug_arr)
                        save_jpg(aug_img, dest_dir / f"{cls}_train_{idx:05d}_{aname}.jpg")
                        n_aug += 1
                    except Exception as aexc:
                        print(f"    [WARN] aug {aname} / {src.name}: {aexc}")

            except Exception as exc:
                print(f"    [WARN] train/{cls}/{src.name}: {exc}")

        train_orig_count[cls]  = n_orig
        train_total_count[cls] = n_orig + n_aug
        print(
            f"        {cls:<12}: {n_orig} originals  "
            f"+ {n_aug} augmented  = {n_orig + n_aug} total"
        )

    # ── STEP 9: Final report ──────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("STEP 9  FINAL REPORT")
    print("=" * 68)
    hdr = f"{'Class':<12} {'Raw':>6}  {'Train':>6}  {'Val':>5}  {'Test':>5}  {'Train+Aug':>10}"
    print(hdr)
    print("-" * 68)

    t_raw = t_train = t_val = t_test = t_aug = 0
    for cls in CLASSES:
        r_raw   = len(raw[cls])
        r_train = train_orig_count[cls]
        r_val   = sum(1 for _ in val_list[cls])
        r_test  = sum(1 for _ in test_list[cls])
        r_aug   = train_total_count[cls]
        print(
            f"{cls:<12} {r_raw:>6}  {r_train:>6}  {r_val:>5}  {r_test:>5}  {r_aug:>10}"
        )
        t_raw += r_raw; t_train += r_train
        t_val += r_val; t_test += r_test; t_aug += r_aug

    print("-" * 68)
    print(
        f"{'TOTAL':<12} {t_raw:>6}  {t_train:>6}  {t_val:>5}  {t_test:>5}  {t_aug:>10}"
    )

    tc = list(train_orig_count.values())
    balance_ratio = (min(tc) / max(tc)) if max(tc) > 0 else 0.0
    print(f"\nBalance ratio : {balance_ratio:.2f}  (should be 1.00)")
    print("=" * 68)
    print("Build complete.")


if __name__ == "__main__":
    main()
