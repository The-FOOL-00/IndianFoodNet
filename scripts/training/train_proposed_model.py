"""
train_proposed_model.py
=======================
Proposed model: EfficientNet-B0 with two-stage fine-tuning for
automated cooking method detection in Indian food images.

Methodology (for paper):
  - Stage 1 (Feature Extraction, epochs 1-10):
      Freeze backbone, train classifier head only with lr=0.001
  - Stage 2 (Full Fine-Tuning, epochs 11-30):
      Unfreeze all layers except BatchNorm, train end-to-end with lr=0.0001
  - Batch Normalization layers kept frozen throughout Stage 2 to preserve
      ImageNet statistics and stabilise training on small dataset
  - ReduceLROnPlateau scheduler to handle learning rate adaptation
  - Dropout(0.3) before final FC layer to reduce overfitting
  - Class-weighted cross-entropy loss to handle class imbalance
  - Early stopping (patience=10) during Stage 2
"""

import os
import json
import time
import shutil
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")



# =============================================================================
# PATHS (Google Colab)
# =============================================================================
DATA_DIR    = "/content/data/processed"
MODEL_DIR   = "/content/models/efficientnet_v2"
METRICS_DIR = "/content/results/metrics"
PLOTS_DIR   = "/content/results/plots"
DRIVE_DIR   = "/content/drive/MyDrive/food_research"
BEST_MODEL  = "/content/models/efficientnet_v2/best_model.pth"

BASELINE_RESULTS = "/content/drive/MyDrive/food_research/results/baseline_results.json"

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
CLASSES        = ["fried", "grilled", "steamed"]
NUM_CLASSES    = len(CLASSES)
IMG_SIZE       = 224
BATCH_SIZE     = 32
STAGE1_EPOCHS  = 10       # Feature extraction
STAGE2_EPOCHS  = 20       # Fine-tuning  (total 30 epochs max)
EARLY_STOP_PAT = 10       # Patience for early stopping in Stage 2
LR_STAGE1      = 1e-3     # Higher lr — only classifier trained
LR_STAGE2      = 1e-4     # Lower lr — full network
DROPOUT_RATE   = 0.3      # Before final FC layer (paper: Section 3.2)
SEED           = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
torch.manual_seed(SEED)
np.random.seed(SEED)


# =============================================================================
# 1. DATA LOADING
# =============================================================================
def get_transforms():
    """
    Training augmentation (paper: Section 3.1 — Data Preprocessing):
      - RandomResizedCrop: simulate varying distances/angles of food shots
      - RandomHorizontalFlip: cooking method invariant to flip
      - ColorJitter: account for varying lighting in food photography
      - Normalise with ImageNet statistics (backbone pretrained on ImageNet)
    Validation/Test: only resize + centre-crop + normalise (no augmentation)
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    return train_tf, eval_tf


def load_datasets():
    train_tf, eval_tf = get_transforms()

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   eval_tf)
    test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  eval_tf)

    # Class weights to handle imbalance (paper: Section 3.3)
    class_counts = Counter(train_ds.targets)
    total        = sum(class_counts.values())
    weights      = [total / (NUM_CLASSES * class_counts[i])
                    for i in range(NUM_CLASSES)]
    class_weights = torch.FloatTensor(weights).to(device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"[DATA] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"[DATA] Class mapping: {train_ds.class_to_idx}")
    print(f"[DATA] Class weights: {dict(zip(CLASSES, [round(w, 4) for w in weights]))}")

    return train_loader, val_loader, test_loader, class_weights, train_ds.class_to_idx


# =============================================================================
# 2. MODEL DEFINITION
# =============================================================================
def build_model():
    """
    EfficientNet-B0 with custom head (paper: Section 3.2 — Model Architecture):
      - Pretrained on ImageNet (transfer learning baseline)
      - Original classifier replaced with: Dropout(0.3) → Linear(num_classes)
      - Dropout(0.3) reduces overfitting on small food-image dataset
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE, inplace=True),   # paper: dropout=0.3
        nn.Linear(in_features, NUM_CLASSES),
    )

    model = model.to(device)
    return model


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def freeze_backbone(model):
    """Stage 1: freeze everything except the classifier head."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_all_except_batchnorm(model):
    """
    Stage 2: unfreeze all layers BUT keep BatchNorm layers frozen
    (paper: Section 3.2 — this preserves ImageNet BN statistics and prevents
    distribution shift when fine-tuning on a small food-image dataset)
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            # Keep BN frozen: fix running stats and parameters
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = True


def set_bn_eval(model):
    """Call in training loop to keep BN layers in eval mode during Stage 2."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


# =============================================================================
# 3. TRAINING UTILITIES
# =============================================================================
def run_epoch(model, loader, criterion, optimizer, stage, training=True):
    if training:
        model.train()
        if stage == 2:
            # Keep BN layers frozen even when model.train() is called
            set_bn_eval(model)
    else:
        model.eval()

    running_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds      = torch.max(outputs, 1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


def backup_to_drive(src, dst_dir, filename=None):
    """Copy a file to Google Drive, silently skip if Drive not mounted."""
    try:
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, filename or os.path.basename(src))
        shutil.copy(src, dst)
        print(f"[DRIVE] Backed up → {dst}")
    except Exception as e:
        print(f"[DRIVE] Skipped backup ({e})")


# =============================================================================
# 4. MAIN TRAINING LOOP
# =============================================================================
def train_model():
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    train_loader, val_loader, test_loader, class_weights, class_to_idx = load_datasets()
    model     = build_model()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "stage_boundary": STAGE1_EPOCHS,
    }

    best_val_acc    = 0.0
    early_stop_ctr  = 0
    total_start     = time.time()

    # -------------------------------------------------------------------------
    # STAGE 1: Feature Extraction (epochs 1–10)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 1 — Feature Extraction (epochs 1-{})".format(STAGE1_EPOCHS))
    print("  Frozen: backbone | Trainable: classifier head only")
    print("  Optimizer: Adam  | LR: {}".format(LR_STAGE1))
    print("=" * 60)

    freeze_backbone(model)
    _, trainable_s1 = count_parameters(model)
    print(f"[STAGE1] Trainable parameters: {trainable_s1:,}")

    optimizer_s1  = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_STAGE1
    )
    scheduler_s1  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_s1, mode="min", factor=0.5, patience=3
    )

    for epoch in range(1, STAGE1_EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer_s1, stage=1, training=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion,
                                    optimizer_s1, stage=1, training=False)
        scheduler_s1.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(f"  Epoch {epoch:02d}/{STAGE1_EPOCHS} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                "epoch":      epoch,
                "stage":      1,
                "model_state_dict": model.state_dict(),
                "val_acc":    best_val_acc,
                "class_to_idx": class_to_idx,
            }, BEST_MODEL)
            print(f"  [CKPT] Best model saved (val_acc={best_val_acc:.4f})")
            backup_to_drive(BEST_MODEL,
                            os.path.join(DRIVE_DIR, "models"),
                            "best_model_efficientnet_v2.pth")

    # -------------------------------------------------------------------------
    # STAGE 2: Full Fine-Tuning (epochs 11–30)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 2 — Full Fine-Tuning (epochs {}-{})".format(
        STAGE1_EPOCHS + 1, STAGE1_EPOCHS + STAGE2_EPOCHS))
    print("  Unfrozen: all layers  |  BatchNorm: KEPT FROZEN")
    print("  Optimizer: Adam  | LR: {}".format(LR_STAGE2))
    print("  Early stopping patience: {}".format(EARLY_STOP_PAT))
    print("=" * 60)

    # Unfreeze all except BatchNorm (paper methodology)
    unfreeze_all_except_batchnorm(model)
    total_s2, trainable_s2 = count_parameters(model)
    print(f"[STAGE2] Total parameters: {total_s2:,} | Trainable: {trainable_s2:,}")

    optimizer_s2 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_STAGE2
    )
    # ReduceLROnPlateau: halve lr if val_loss stagnates for 5 epochs
    scheduler_s2 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_s2, mode="min", factor=0.5, patience=5
    )

    best_val_acc_s2 = best_val_acc  # carry over from Stage 1

    for epoch in range(1, STAGE2_EPOCHS + 1):
        global_epoch = STAGE1_EPOCHS + epoch

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer_s2, stage=2, training=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion,
                                    optimizer_s2, stage=2, training=False)
        scheduler_s2.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(f"  Epoch {global_epoch:02d}/{STAGE1_EPOCHS + STAGE2_EPOCHS} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}")

        if vl_acc > best_val_acc_s2:
            best_val_acc_s2 = vl_acc
            early_stop_ctr  = 0
            torch.save({
                "epoch":      global_epoch,
                "stage":      2,
                "model_state_dict": model.state_dict(),
                "val_acc":    best_val_acc_s2,
                "class_to_idx": class_to_idx,
            }, BEST_MODEL)
            print(f"  [CKPT] Best model saved (val_acc={best_val_acc_s2:.4f})")
            backup_to_drive(BEST_MODEL,
                            os.path.join(DRIVE_DIR, "models"),
                            "best_model_efficientnet_v2.pth")
        else:
            early_stop_ctr += 1
            print(f"  [EARLY STOP] No improvement for {early_stop_ctr}/{EARLY_STOP_PAT} epochs")
            if early_stop_ctr >= EARLY_STOP_PAT:
                print(f"\n[EARLY STOP] Triggered at epoch {global_epoch}. "
                      f"Best val_acc={best_val_acc_s2:.4f}")
                break

    training_time = time.time() - total_start
    print(f"\n[INFO] Total training time: {training_time:.1f}s "
          f"({training_time/60:.1f} min)")

    return model, history, test_loader, class_to_idx, training_time, total_s2


# =============================================================================
# 5. TEST SET EVALUATION
# =============================================================================
def evaluate_on_test(model, test_loader, class_to_idx, training_time, total_params):
    print("\n[EVAL] Running evaluation on test set...")

    # Load the best checkpoint
    checkpoint = torch.load(BEST_MODEL, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs  = model(images)
            probs    = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=list(range(NUM_CLASSES))
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro"
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )

    cm = confusion_matrix(all_labels, all_preds)

    per_class = {}
    for i, cls in enumerate(CLASSES):
        per_class[cls] = {
            "precision": round(float(precision[i]), 4),
            "recall":    round(float(recall[i]),    4),
            "f1_score":  round(float(f1[i]),        4),
            "support":   int(support[i]),
        }

    results = {
        "model":          "EfficientNet-B0 (Two-Stage Fine-Tuning)",
        "timestamp":      datetime.now().isoformat(),
        "best_epoch":     int(checkpoint["epoch"]),
        "best_val_acc":   round(float(checkpoint["val_acc"]), 4),
        "test_accuracy":  round(float(accuracy), 4),
        "macro_precision": round(float(p_macro),    4),
        "macro_recall":   round(float(r_macro),     4),
        "macro_f1":       round(float(f1_macro),    4),
        "weighted_precision": round(float(p_weighted), 4),
        "weighted_recall":   round(float(r_weighted),  4),
        "weighted_f1":       round(float(f1_weighted), 4),
        "per_class_metrics": per_class,
        "confusion_matrix":  cm.tolist(),
        "training_time_seconds": round(training_time, 2),
        "training_time_minutes": round(training_time / 60, 2),
        "total_parameters":  total_params,
        "hyperparameters": {
            "batch_size":       BATCH_SIZE,
            "stage1_epochs":    STAGE1_EPOCHS,
            "stage2_epochs":    STAGE2_EPOCHS,
            "lr_stage1":        LR_STAGE1,
            "lr_stage2":        LR_STAGE2,
            "dropout":          DROPOUT_RATE,
            "early_stop_patience": EARLY_STOP_PAT,
            "img_size":         IMG_SIZE,
        },
    }

    metrics_path = os.path.join(METRICS_DIR, "proposed_results.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[EVAL] Metrics saved → {metrics_path}")

    # Backup metrics to Drive
    backup_to_drive(metrics_path,
                    os.path.join(DRIVE_DIR, "results"),
                    "proposed_results.json")

    print(f"\n[EVAL] Test Accuracy : {accuracy:.4f}")
    print(f"[EVAL] Macro F1      : {f1_macro:.4f}")
    print(f"[EVAL] Weighted F1   : {f1_weighted:.4f}")
    print(classification_report(all_labels, all_preds,
                                target_names=CLASSES, digits=4))

    return results, cm, all_labels, all_preds


# =============================================================================
# 6. PLOTS
# =============================================================================
def plot_accuracy(history):
    epochs = range(1, len(history["train_acc"]) + 1)
    boundary = history["stage_boundary"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_acc"], "b-o", markersize=4, label="Train Accuracy")
    ax.plot(epochs, history["val_acc"],   "r-o", markersize=4, label="Val Accuracy")
    ax.axvline(x=boundary + 0.5, color="green", linestyle="--",
               linewidth=1.5, label="Stage 1 → Stage 2")
    ax.axvspan(1, boundary + 0.5, alpha=0.05, color="blue",  label="Stage 1 (Feature Extraction)")
    ax.axvspan(boundary + 0.5, max(epochs) + 0.5, alpha=0.05,
               color="red", label="Stage 2 (Fine-Tuning)")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("EfficientNet-B0: Training & Validation Accuracy\n"
                 "(Two-Stage Fine-Tuning)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "proposed_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Accuracy plot → {path}")
    backup_to_drive(path, os.path.join(DRIVE_DIR, "results", "plots"))
    return path


def plot_loss(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    boundary = history["stage_boundary"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
    ax.plot(epochs, history["val_loss"],   "r-o", markersize=4, label="Val Loss")
    ax.axvline(x=boundary + 0.5, color="green", linestyle="--",
               linewidth=1.5, label="Stage 1 → Stage 2")
    ax.axvspan(1, boundary + 0.5, alpha=0.05, color="blue")
    ax.axvspan(boundary + 0.5, max(epochs) + 0.5, alpha=0.05, color="red")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title("EfficientNet-B0: Training & Validation Loss\n"
                 "(Two-Stage Fine-Tuning)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "proposed_loss.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Loss plot → {path}")
    backup_to_drive(path, os.path.join(DRIVE_DIR, "results", "plots"))
    return path


def plot_confusion_matrix(cm):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("True Label",      fontsize=11)

    # Normalised
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[1],
                vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix (Normalised)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].set_ylabel("True Label",      fontsize=11)

    fig.suptitle("EfficientNet-B0 — Test Set Confusion Matrix",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, "proposed_confusion.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Confusion matrix → {path}")
    backup_to_drive(path, os.path.join(DRIVE_DIR, "results", "plots"))
    return path


# =============================================================================
# 7. COMPARISON TABLE
# =============================================================================
def print_comparison_table(proposed_results):
    print("\n" + "=" * 70)
    print("  BASELINE vs PROPOSED MODEL — COMPARISON TABLE")
    print("=" * 70)

    # Load baseline results from Drive
    baseline = None
    try:
        with open(BASELINE_RESULTS) as f:
            baseline = json.load(f)
        print(f"[TABLE] Loaded baseline results from {BASELINE_RESULTS}")
    except FileNotFoundError:
        print(f"[TABLE] Baseline file not found at {BASELINE_RESULTS}. "
              "Using placeholder values.")
    except Exception as e:
        print(f"[TABLE] Could not load baseline ({e}). Using placeholder values.")

    # Extract proposed values
    p_acc   = proposed_results["test_accuracy"]
    p_f1    = proposed_results["weighted_f1"]
    p_time  = proposed_results["training_time_minutes"]
    p_total = proposed_results["total_parameters"]

    # Baseline values (real or placeholder)
    if baseline:
        b_acc   = baseline.get("test_accuracy",          "N/A")
        b_f1    = baseline.get("weighted_f1",            baseline.get("macro_f1", "N/A"))
        b_time  = baseline.get("training_time_minutes",  "N/A")
        b_total = baseline.get("total_parameters",       "N/A")
        b_model = baseline.get("model",                  "Baseline")
    else:
        b_acc   = "—"
        b_f1    = "—"
        b_time  = "—"
        b_total = "—"
        b_model = "Baseline (not found)"

    def fmt(v):
        if isinstance(v, float): return f"{v:.4f}"
        if isinstance(v, int):   return f"{v:,}"
        return str(v)

    col_w = 28
    row_fmt = f"  {{:<{col_w}}}  {{:<{col_w}}}  {{:<{col_w}}}"

    print(row_fmt.format("Metric", b_model[:col_w], "EfficientNet-B0 (Proposed)"))
    print("  " + "-" * (col_w * 3 + 4))
    print(row_fmt.format("Accuracy",          fmt(b_acc),   fmt(p_acc)))
    print(row_fmt.format("Weighted F1-Score", fmt(b_f1),    fmt(p_f1)))
    print(row_fmt.format("Training Time (min)", fmt(b_time), fmt(p_time)))
    print(row_fmt.format("Total Parameters",  fmt(b_total), fmt(p_total)))

    # Delta row (only when baseline is numeric)
    if baseline and isinstance(b_acc, (int, float)) and isinstance(p_acc, float):
        delta_acc = p_acc - b_acc
        delta_f1  = p_f1  - (b_f1 if isinstance(b_f1, float) else 0)
        print("  " + "-" * (col_w * 3 + 4))
        print(row_fmt.format(
            "Accuracy Δ (proposed−base)",
            "",
            f"{delta_acc:+.4f}"
        ))
        print(row_fmt.format(
            "Weighted F1 Δ",
            "",
            f"{delta_f1:+.4f}"
        ))

    print("=" * 70)
    print("  Note: Two-stage training — Stage 1 (epochs 1-10) feature")
    print("        extraction; Stage 2 (epochs 11-30) full fine-tuning")
    print("        with BatchNorm layers frozen throughout Stage 2.")
    print("=" * 70 + "\n")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Proposed Model: EfficientNet-B0, Two-Stage Fine-Tuning")
    print("  Indian Food Cooking Method Detection")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Train
    model, history, test_loader, class_to_idx, training_time, total_params = train_model()

    # Evaluate
    results, cm, all_labels, all_preds = evaluate_on_test(
        model, test_loader, class_to_idx, training_time, total_params
    )

    # Plots
    plot_accuracy(history)
    plot_loss(history)
    plot_confusion_matrix(cm)

    # Comparison table
    print_comparison_table(results)

    print(f"[DONE] All outputs saved.")
    print(f"       Model    : {BEST_MODEL}")
    print(f"       Metrics  : {METRICS_DIR}/proposed_results.json")
    print(f"       Plots    : {PLOTS_DIR}/proposed_accuracy.png")
    print(f"                  {PLOTS_DIR}/proposed_loss.png")
    print(f"                  {PLOTS_DIR}/proposed_confusion.png")
    print(f"       Drive    : {DRIVE_DIR}/")
