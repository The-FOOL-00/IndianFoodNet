# Loads all model results JSONs, prints a full comparison table, and saves a
# master comparison chart to results/plots/all_models_comparison.png.

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Config ───────────────────────────────────────────────────────────────────
METRICS_DIR = "/content/results/metrics"
PLOTS_DIR   = "/content/results/plots"
DRIVE_DIR   = "/content/drive/MyDrive/food_research"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(f"{DRIVE_DIR}/results", exist_ok=True)

CLASS_NAMES = ["fried", "grilled", "steamed"]

# Each entry: (display_name, json_filename)
# The loader handles both JSON shapes written by train_baseline.py /
# train_*.py  (key "overall_accuracy") and train_proposed_model.py
# (key "test_accuracy").
MODEL_REGISTRY = [
    ("ResNet-50",        "baseline_results.json"),
    ("MobileNetV2",      "mobilenetv2_results.json"),
    ("InceptionV3",      "inceptionv3_results.json"),
    ("DenseNet-121",     "densenet121_results.json"),
    ("EfficientNet-B0",  "proposed_results.json"),
]

# ─── Load results ─────────────────────────────────────────────────────────────
def load_result(json_path):
    with open(json_path) as f:
        d = json.load(f)
    acc = d.get("overall_accuracy") or d.get("test_accuracy", float("nan"))
    pcm = d.get("per_class_metrics", {})
    f1_per_class = {cls: pcm.get(cls, {}).get("f1_score", float("nan"))
                    for cls in CLASS_NAMES}
    macro_f1 = float(np.nanmean(list(f1_per_class.values())))
    return {"accuracy": acc, "f1_per_class": f1_per_class, "macro_f1": macro_f1}


rows = []
for display_name, fname in MODEL_REGISTRY:
    path = os.path.join(METRICS_DIR, fname)
    if not os.path.isfile(path):
        print(f"  [skip] {fname} not found — run the corresponding training script first.")
        continue
    data = load_result(path)
    rows.append((display_name, data))

if not rows:
    raise SystemExit("No result files found. Run training scripts first.")

# ─── Print comparison table ───────────────────────────────────────────────────
col_w = 14
sep   = "=" * (col_w + 1 + 10 + 1 + 8 + 1 + 8 + 1 + 8 + 1 + 10)
header = (
    f"{'Model':<{col_w}} {'Accuracy':>10} {'F1-fried':>8} "
    f"{'F1-grld':>8} {'F1-stmd':>8} {'MacroF1':>10}"
)
print(sep)
print("  ALL MODELS COMPARISON")
print(sep)
print("  " + header)
print("  " + "-" * len(header))
for name, d in rows:
    f1 = d["f1_per_class"]
    print(
        f"  {name:<{col_w}} "
        f"{d['accuracy']:>10.4f} "
        f"{f1['fried']:>8.4f} "
        f"{f1['grilled']:>8.4f} "
        f"{f1['steamed']:>8.4f} "
        f"{d['macro_f1']:>10.4f}"
    )
print(sep)

# ─── Comparison chart ─────────────────────────────────────────────────────────
model_names  = [r[0] for r in rows]
accuracies   = [r[1]["accuracy"] for r in rows]
macro_f1s    = [r[1]["macro_f1"] for r in rows]
f1_fried     = [r[1]["f1_per_class"]["fried"]   for r in rows]
f1_grilled   = [r[1]["f1_per_class"]["grilled"] for r in rows]
f1_steamed   = [r[1]["f1_per_class"]["steamed"] for r in rows]

n      = len(rows)
x      = np.arange(n)
width  = 0.15
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#937860"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Food Classification — All Models Comparison", fontsize=14, fontweight="bold")

# Left panel: accuracy + macro-F1
ax = axes[0]
ax.bar(x - width / 2, accuracies, width, label="Accuracy", color=colors[0], alpha=0.85)
ax.bar(x + width / 2, macro_f1s,  width, label="Macro F1",  color=colors[1], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
ax.set_ylabel("Score")
ax.set_title("Overall Accuracy & Macro F1")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
for i, (a, f) in enumerate(zip(accuracies, macro_f1s)):
    ax.text(i - width / 2, a + 0.005, f"{a:.3f}", ha="center", va="bottom", fontsize=7)
    ax.text(i + width / 2, f + 0.005, f"{f:.3f}", ha="center", va="bottom", fontsize=7)

# Right panel: per-class F1
ax = axes[1]
ax.bar(x - width,     f1_fried,   width, label="F1 fried",   color=colors[2], alpha=0.85)
ax.bar(x,             f1_grilled, width, label="F1 grilled",  color=colors[3], alpha=0.85)
ax.bar(x + width,     f1_steamed, width, label="F1 steamed",  color=colors[4], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
ax.set_ylabel("F1 Score")
ax.set_title("Per-Class F1 Score")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
for i in range(n):
    for offset, vals in [(-width, f1_fried), (0, f1_grilled), (width, f1_steamed)]:
        ax.text(i + offset, vals[i] + 0.005, f"{vals[i]:.3f}",
                ha="center", va="bottom", fontsize=6.5)

plt.tight_layout()
out_path = os.path.join(PLOTS_DIR, "all_models_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nComparison chart saved → {out_path}")

# ─── Render a text table as an image too ─────────────────────────────────────
col_labels = ["Model", "Accuracy", "F1 fried", "F1 grilled", "F1 steamed", "Macro F1"]
table_data = []
for name, d in rows:
    f1 = d["f1_per_class"]
    table_data.append([
        name,
        f"{d['accuracy']:.4f}",
        f"{f1['fried']:.4f}",
        f"{f1['grilled']:.4f}",
        f"{f1['steamed']:.4f}",
        f"{d['macro_f1']:.4f}",
    ])

fig_t, ax_t = plt.subplots(figsize=(11, 1.2 + 0.5 * len(rows)))
ax_t.axis("off")
tbl = ax_t.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.6)

# Highlight header row
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#2C5F8A")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Highlight the best accuracy row
best_idx = int(np.argmax([r[1]["accuracy"] for r in rows]))
for j in range(len(col_labels)):
    tbl[best_idx + 1, j].set_facecolor("#D4EDDA")

ax_t.set_title("Food Classification — Model Comparison Table",
               fontsize=12, fontweight="bold", pad=10)
plt.tight_layout()
table_path = os.path.join(PLOTS_DIR, "all_models_comparison_table.png")
plt.savefig(table_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Comparison table image → {table_path}")

# ─── Back up to Drive ─────────────────────────────────────────────────────────
import shutil
shutil.copy(out_path,    f"{DRIVE_DIR}/results/all_models_comparison.png")
shutil.copy(table_path,  f"{DRIVE_DIR}/results/all_models_comparison_table.png")
print("Comparison charts backed up to Google Drive successfully")
