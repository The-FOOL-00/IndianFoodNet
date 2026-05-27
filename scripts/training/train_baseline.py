# This is the BASELINE model. It uses ResNet-50 with frozen pretrained weights and only trains
# the final classification layer. This represents the standard approach used in existing literature
# and serves as the comparison point for our proposed EfficientNet-B0 model.

import os
import json
import time
import shutil
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR    = "/content/data/processed"
MODEL_DIR   = "/content/models/baseline"
METRICS_DIR = "/content/results/metrics"
PLOTS_DIR   = "/content/results/plots"
DRIVE_DIR   = "/content/drive/MyDrive/food_research"
BEST_MODEL    = os.path.join(MODEL_DIR, "best_model.pth")

NUM_CLASSES   = 3
BATCH_SIZE    = 32
NUM_EPOCHS    = 20
LEARNING_RATE = 1e-3
CLASS_NAMES   = ["fried", "grilled", "steamed"]   # ImageFolder sorts alphabetically

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Directory setup ──────────────────────────────────────────────────────────
for d in (MODEL_DIR, METRICS_DIR, PLOTS_DIR,
          f"{DRIVE_DIR}/models", f"{DRIVE_DIR}/results"):
    os.makedirs(d, exist_ok=True)

# ─── Data transforms ──────────────────────────────────────────────────────────
_mean = [0.485, 0.456, 0.406]
_std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_mean, _std),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_mean, _std),
])

# ─── Datasets & loaders ───────────────────────────────────────────────────────
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=eval_transform)
test_dataset  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# Use the actual class order from ImageFolder (alphabetical by folder name)
CLASS_NAMES = train_dataset.classes
print(f"Classes (in order): {CLASS_NAMES}")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
print(f"Device: {DEVICE}\n")

# ─── Model ────────────────────────────────────────────────────────────────────
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final FC layer — only this layer will be trained
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model = model.to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters : {trainable:,}  /  Total: {total:,}\n")

# ─── Loss & optimizer ─────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# ─── Training loop ────────────────────────────────────────────────────────────
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

best_val_acc  = 0.0
training_start = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    # ── Train ──
    model.train()
    running_loss, running_correct = 0.0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss    += loss.item() * images.size(0)
        running_correct += (outputs.argmax(1) == labels).sum().item()

    train_loss = running_loss / len(train_dataset)
    train_acc  = running_correct / len(train_dataset)

    # ── Validate ──
    model.eval()
    val_loss, val_correct = 0.0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            val_loss    += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_loss = val_loss / len(val_dataset)
    val_acc  = val_correct / len(val_dataset)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    improved = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), BEST_MODEL)
        shutil.copy(BEST_MODEL,
                    f"{DRIVE_DIR}/models/best_model_baseline.pth")
        improved = "  ✓ saved+backed up"

    print(
        f"Epoch [{epoch:02d}/{NUM_EPOCHS}]  "
        f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
        f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}{improved}"
    )

training_time = time.time() - training_start
print(f"\nTraining complete in {training_time:.1f}s  |  Best Val Acc: {best_val_acc:.4f}\n")

# ─── Test evaluation (best checkpoint) ────────────────────────────────────────
model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        preds  = model(images).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

overall_acc = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, labels=list(range(NUM_CLASSES)), zero_division=0
)
cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))

per_class = {
    CLASS_NAMES[i]: {
        "precision": round(float(precision[i]), 4),
        "recall":    round(float(recall[i]),    4),
        "f1_score":  round(float(f1[i]),        4),
    }
    for i in range(NUM_CLASSES)
}

results = {
    "model":           "ResNet-50 Baseline (frozen backbone)",
    "overall_accuracy": round(float(overall_acc), 4),
    "per_class_metrics": per_class,
    "confusion_matrix":  cm.tolist(),
    "training_time_seconds": round(training_time, 2),
    "best_val_accuracy":     round(float(best_val_acc), 4),
    "num_epochs":            NUM_EPOCHS,
    "batch_size":            BATCH_SIZE,
    "learning_rate":         LEARNING_RATE,
    "class_names":           CLASS_NAMES,
}

metrics_path = os.path.join(METRICS_DIR, "baseline_results.json")
with open(metrics_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Metrics saved → {metrics_path}")

# ─── Plots ────────────────────────────────────────────────────────────────────
epochs_range = range(1, NUM_EPOCHS + 1)

# Accuracy curve
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["train_acc"], marker="o", label="Train Accuracy")
plt.plot(epochs_range, history["val_acc"],   marker="s", label="Val Accuracy")
plt.title("ResNet-50 Baseline — Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
acc_plot = os.path.join(PLOTS_DIR, "baseline_accuracy.png")
plt.savefig(acc_plot, dpi=150)
plt.close()
print(f"Accuracy plot saved → {acc_plot}")

# Loss curve
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["train_loss"], marker="o", label="Train Loss")
plt.plot(epochs_range, history["val_loss"],   marker="s", label="Val Loss")
plt.title("ResNet-50 Baseline — Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
loss_plot = os.path.join(PLOTS_DIR, "baseline_loss.png")
plt.savefig(loss_plot, dpi=150)
plt.close()
print(f"Loss plot saved     → {loss_plot}")

# Confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    linewidths=0.5,
)
plt.title("ResNet-50 Baseline — Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
cm_plot = os.path.join(PLOTS_DIR, "baseline_confusion.png")
plt.savefig(cm_plot, dpi=150)
plt.close()
print(f"Confusion matrix    → {cm_plot}\n")

# ─── Final summary ────────────────────────────────────────────────────────────
sep = "=" * 55
print(sep)
print("  BASELINE MODEL — FINAL RESULTS")
print(sep)
print(f"  Model          : ResNet-50 (frozen backbone)")
print(f"  Overall Acc    : {overall_acc:.4f}  ({overall_acc*100:.2f}%)")
print(f"  Training time  : {training_time:.1f}s")
print(f"  Best Val Acc   : {best_val_acc:.4f}  ({best_val_acc*100:.2f}%)")
print(sep)
print(f"  {'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*8}")
for cls in CLASS_NAMES:
    m = per_class[cls]
    print(f"  {cls:<12} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1_score']:>8.4f}")
print(sep)
print(f"  Confusion matrix (rows=true, cols=pred):")
header = "  " + " " * 10 + "  ".join(f"{c:>8}" for c in CLASS_NAMES)
print(header)
for i, row in enumerate(cm):
    row_str = "  ".join(f"{v:>8}" for v in row)
    print(f"  {CLASS_NAMES[i]:<10}{row_str}")
print(sep)
print(f"  Saved model    : {BEST_MODEL}")
print(f"  Saved metrics  : {metrics_path}")
print(sep)
shutil.copy(metrics_path,
            f"{DRIVE_DIR}/results/baseline_results.json")
for plot in ["baseline_accuracy.png",
             "baseline_loss.png",
             "baseline_confusion.png"]:
    shutil.copy(f"{PLOTS_DIR}/{plot}",
                f"{DRIVE_DIR}/results/{plot}")
print("All results backed up to Google Drive successfully")
