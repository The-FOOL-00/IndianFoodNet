import os, json, random, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

CALORIE_DB = {
    "fried":   {"average": 295},
    "steamed": {"average": 143},
    "grilled": {"average": 222},
}

CLASS_CALORIES = {
    "fried":   295,
    "steamed": 143,
    "grilled": 222,
}

BASELINE_CALORIES = 200
LABEL_TO_METHOD = {0: "fried", 1: "grilled", 2: "steamed"}
MODEL_DST  = "/content/best_model_efficientnet.pth"
RESULTS_DIR = "/content/drive/MyDrive/food_research/results"
TEST_DIR   = "/content/data/processed/test"
DEVICE     = torch.device("cpu")

def build_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 3),
    )
    return model

def load_model(path):
    model = build_model()
    checkpoint = torch.load(path, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict",
                 checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded")
    return model

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

def run_experiment():
    random.seed(42)
    model = load_model(MODEL_DST)
    baseline_errors = []
    proposed_errors = []
    records = []

    for true_method in ["fried", "steamed", "grilled"]:
        folder = os.path.join(TEST_DIR, true_method)
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg",".jpeg",".png"))]
        chosen = random.sample(files, 10)
        gt_cal = CLASS_CALORIES[true_method]

        for fname in chosen:
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert("RGB")
            tensor = TRANSFORM(img).unsqueeze(0)
            with torch.no_grad():
                logits = model(tensor)
            pred_label = int(torch.argmax(logits, dim=1).item())
            pred_method = LABEL_TO_METHOD[pred_label]
            proposed_cal = CLASS_CALORIES[pred_method]
            baseline_cal = BASELINE_CALORIES
            b_err = abs(baseline_cal - gt_cal)
            p_err = abs(proposed_cal - gt_cal)
            baseline_errors.append(b_err)
            proposed_errors.append(p_err)
            records.append({
                "filename": fname,
                "true_method": true_method,
                "predicted_method": pred_method,
                "correct": pred_method == true_method,
                "ground_truth_cal": gt_cal,
                "baseline_cal": baseline_cal,
                "proposed_cal": proposed_cal,
                "baseline_error": b_err,
                "proposed_error": p_err,
            })

    b_arr = np.array(baseline_errors, dtype=float)
    p_arr = np.array(proposed_errors, dtype=float)
    baseline_mae = float(np.mean(b_arr))
    proposed_mae = float(np.mean(p_arr))
    improvement  = (baseline_mae - proposed_mae) / baseline_mae * 100
    correct_preds = sum(1 for r in records if r["correct"])
    accuracy = correct_preds / len(records) * 100

    print("\n" + "="*50)
    print(f"Samples tested    : {len(records)}")
    print(f"Model accuracy    : {accuracy:.1f}%")
    print(f"Baseline MAE      : {baseline_mae:.2f} kcal")
    print(f"Proposed MAE      : {proposed_mae:.2f} kcal")
    print(f"Improvement       : {improvement:.2f}%")
    print("="*50)

    methods = ["fried", "steamed", "grilled"]
    b_per_class = [np.mean([r["baseline_error"]
                   for r in records if r["true_method"]==m])
                   for m in methods]
    p_per_class = [np.mean([r["proposed_error"]
                   for r in records if r["true_method"]==m])
                   for m in methods]

    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars_b = ax.bar(x - width/2, b_per_class, width,
                    label="Baseline (method-agnostic)",
                    color="#E74C3C", alpha=0.85)
    bars_p = ax.bar(x + width/2, p_per_class, width,
                    label="Proposed (method-aware)",
                    color="#2ECC71", alpha=0.85)
    ax.set_xlabel("Cooking Method Class", fontsize=12)
    ax.set_ylabel("Mean Calorie Error (kcal)", fontsize=12)
    ax.set_title("Calorie Estimation Error:\nBaseline vs Proposed System",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Fried","Steamed","Grilled"], fontsize=12)
    ax.legend(fontsize=11)
    ax.bar_label(bars_b, fmt="%.1f", padding=3)
    ax.bar_label(bars_p, fmt="%.1f", padding=3)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    textstr = (f"Baseline MAE: {baseline_mae:.1f} kcal\n"
               f"Proposed MAE: {proposed_mae:.1f} kcal\n"
               f"Improvement: {improvement:.1f}%")
    props = dict(boxstyle="round", facecolor="lightyellow", alpha=0.8)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes,
            fontsize=10, va="top", ha="right", bbox=props)
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig_path = os.path.join(RESULTS_DIR, "proposed_calorie_comparison.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    json_path = os.path.join(RESULTS_DIR, "calorie_experiment.json")
    with open(json_path, "w") as f:
        json.dump({"summary": {
            "n_samples": len(records),
            "model_accuracy_pct": round(accuracy, 2),
            "baseline_mae": round(baseline_mae, 4),
            "proposed_mae": round(proposed_mae, 4),
            "improvement_pct": round(improvement, 4),
        }, "details": records}, f, indent=2)
    return records

records = run_experiment()
