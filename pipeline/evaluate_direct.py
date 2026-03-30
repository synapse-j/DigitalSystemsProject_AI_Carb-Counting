"""
pipeline/evaluate_direct.py
============================
Evaluates the trained ResNet-50 on the final_eval test set
(Food-101 TEST split - genuinely unseen during training).

Uses Test-Time Augmentation (TTA): averages softmax probabilities over
N augmented views of each image. Typically adds +1-3% exact accuracy
for free - no retraining required.

Metrics reported:
  - Exact accuracy
  - Clinically acceptable accuracy (±1 range / ±20g)
  - Dangerous predictions (±2+ ranges / ±40g)
  - Per-range accuracy breakdown
  - Confusion matrix
"""

# ==================================================================
# IMPORTS
# ==================================================================

# Standard library
import json
import os
import sys
from pathlib import Path

# Extend the Python path so local modules (config, pipeline) are
# importable when this script is run directly rather than as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Third-party
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms
from torchvision.models import resnet50

# Local
from config import (
    DATASET_DIR, RESULTS_DIR, DEVICE,
    NUM_CARB_RANGES, CARB_RANGE_LABELS,
)
from pipeline.train_direct import CarbRangeDataset, MODEL_DIR

# ==================================================================
# CONSTANTS
# ==================================================================

# Human-readable carb range labels used in plots and printed reports
RANGE_LABELS = list(CARB_RANGE_LABELS.values())

# ImageNet normalisation statistics - must match the values used during
# training so the model receives inputs from the same distribution.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Deterministic transform for the base evaluation pass.
# Centre-crop is reproducible and avoids cutting off food at the image edges.
VAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Stochastic transform for the TTA augmentation passes.
# Random crop + flip + colour jitter simulate natural variation in how
# a user might photograph the same meal (angle, lighting, framing).
TTA_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ==================================================================
# MODEL LOADING
# ==================================================================

def load_model() -> nn.Module:
    """
    Reconstruct the ResNet-50 architecture and load the saved checkpoint.

    The classifier head must exactly match what was used during training:
      Linear(2048 -> 512) -> BatchNorm -> ReLU -> Dropout(0.4) -> Linear(512 -> 5)
    """
    model = resnet50(weights=None)  # weights=None because we load from checkpoint below
    in_features = model.fc.in_features  # ResNet-50 backbone outputs 2048 features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(512, NUM_CARB_RANGES),
    )
    checkpoint_path = os.path.join(MODEL_DIR, "resnet50_best.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model not found at {checkpoint_path}. "
            "Run 'python run.py --stage train' first."
        )
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    return model.eval().to(DEVICE)


# ==================================================================
# INFERENCE WITH TEST-TIME AUGMENTATION
# ==================================================================

@torch.no_grad()
def run_inference_tta(model: nn.Module, samples: list, n_aug: int = 8):
    """
    Run TTA inference over all test samples.

    For each image, average softmax probabilities over:
      1 × deterministic centre-crop (VAL_TF)
      n_aug × random augmentations  (TTA_TF)

    Averaging over multiple views reduces prediction variance caused by
    ambiguous crops, lighting variation, and partial occlusions - all
    common challenges in real-world Food-101 images.

    Args:
        model:   Trained ResNet-50 in eval mode.
        samples: List of (image_path, true_label) tuples.
        n_aug:   Number of stochastic augmentation passes per image.

    Returns:
        Tuple of (true_labels, pred_labels, confidences) as numpy arrays.
    """
    true_labels, pred_labels, confidences = [], [], []

    model.eval()
    for sample_index, (img_path, label) in enumerate(samples):
        img = Image.open(img_path).convert("RGB")

        # Base deterministic pass - ensures at least one stable prediction
        base_tensor = VAL_TF(img).unsqueeze(0).to(DEVICE)
        prob_sum = torch.softmax(model(base_tensor), dim=1)

        # Accumulate stochastic TTA predictions
        for _ in range(n_aug):
            aug_tensor = TTA_TF(img).unsqueeze(0).to(DEVICE)
            prob_sum += torch.softmax(model(aug_tensor), dim=1)

        # Average over all passes (1 base + n_aug stochastic) to get final probs
        avg_probs = prob_sum / (n_aug + 1)
        confidence, predicted_range = avg_probs.max(dim=1)

        true_labels.append(label)
        pred_labels.append(predicted_range.cpu().item())
        confidences.append(confidence.cpu().item())

        if (sample_index + 1) % 500 == 0:
            running_acc = sum(
                t == p for t, p in zip(true_labels, pred_labels)
            ) / len(true_labels)
            print(f"  [{sample_index+1:5d}/{len(samples):5d}]  running acc={running_acc:.3f}")

    return (
        np.array(true_labels),
        np.array(pred_labels),
        np.array(confidences),
    )


# ==================================================================
# METRICS
# ==================================================================

def compute_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
    """
    Compute evaluation metrics with clinical safety framing.

    Three tiers of prediction quality are reported:
      - Exact:      Correct carb range predicted.
      - Clinical:   Off by ±1 range (±20g). Acceptable per Özkaya et al. (2026)
                    because minor insulin dose adjustments can compensate.
      - Dangerous:  Off by ±2+ ranges (±40g). Clinically unsafe - large enough
                    to risk hypoglycaemia (under-bolus) or hyperglycaemia (over-bolus).
    """
    exact_accuracy    = float((true_labels == pred_labels).mean())

    # ±1 range = ±20g carbs - the clinical tolerance threshold for T1D management
    clinical_accuracy = float((np.abs(true_labels - pred_labels) <= 1).mean())

    # ±2+ ranges = ±40g carbs - errors large enough to cause a medical incident
    dangerous_rate    = float((np.abs(true_labels - pred_labels) >= 2).mean())

    classification_rep = classification_report(
        true_labels, pred_labels, target_names=RANGE_LABELS,
        output_dict=True, zero_division=0,
    )

    # Per-range breakdown: reveals which carb bands are hardest to classify
    per_range_accuracy = {}
    for range_index in range(NUM_CARB_RANGES):
        range_mask = true_labels == range_index
        if range_mask.sum() > 0:
            per_range_accuracy[RANGE_LABELS[range_index]] = round(
                float((true_labels[range_mask] == pred_labels[range_mask]).mean()), 3
            )

    print(f"\n  {'─'*50}")
    print(f"  ResNet-50 + Ordinal Loss + TTA Results")
    print(f"  {'─'*50}")
    print(f"  Exact accuracy:            {exact_accuracy:.3f}  ({exact_accuracy*100:.1f}%)")
    print(f"  Clinical acc (±1 range):   {clinical_accuracy:.3f}  ({clinical_accuracy*100:.1f}%)")
    print(f"  Dangerous preds (±2+):     {dangerous_rate:.3f}  ({dangerous_rate*100:.1f}%)")
    print(f"\n  Per-range accuracy:")
    for range_label, accuracy in per_range_accuracy.items():
        progress_bar = "=" * int(accuracy * 20)
        print(f"    {range_label:<8} {accuracy:.3f}  {progress_bar}")

    return {
        "model":                                    "ResNet-50 (Direct + TTA)",
        "test_accuracy":                            round(exact_accuracy, 4),
        "clinically_acceptable_accuracy_+-1_range": round(clinical_accuracy, 4),
        "dangerous_predictions_+-2_ranges":         round(dangerous_rate, 4),
        "per_range_accuracy":                       per_range_accuracy,
        "classification_report":                    classification_rep,
        "confusion_matrix":
            confusion_matrix(true_labels, pred_labels).tolist(),
    }


# ==================================================================
# VISUALISATION
# ==================================================================

def plot_confusion_matrix(true_labels: np.ndarray, pred_labels: np.ndarray):
    """
    Save a side-by-side confusion matrix figure (raw counts + normalised).

    The normalised matrix is more informative for imbalanced classes: it
    shows true class recall rather than raw counts, making it easier to
    spot which carb ranges are being confused with each other.
    """
    cm_raw        = confusion_matrix(true_labels, pred_labels)
    cm_normalised = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, matrix_data, title, fmt in zip(
        axes,
        [cm_raw, cm_normalised],
        ["ResNet-50 - Counts", "ResNet-50 - Normalised"],
        ["d", ".2f"],
    ):
        heatmap_img = ax.imshow(matrix_data, cmap="Blues")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Range")
        ax.set_ylabel("True Range")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(RANGE_LABELS, rotation=30, ha="right")
        ax.set_yticklabels(RANGE_LABELS)
        plt.colorbar(heatmap_img, ax=ax)

        # Use white text on dark cells and black text on light cells for readability
        cell_threshold = matrix_data.max() / 2
        for row in range(5):
            for col in range(5):
                ax.text(
                    col, row, f"{matrix_data[row, col]:{fmt}}",
                    ha="center", va="center", fontsize=8,
                    color="white" if matrix_data[row, col] > cell_threshold else "black",
                )

    plt.suptitle(
        "Confusion Matrix - ResNet-50 Direct Classifier",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, "cm_resnet50.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  OK Confusion matrix -> {output_path}")


def plot_per_range_accuracy(per_range_accuracy: dict):
    """
    Save a colour-coded bar chart of per-range classification accuracy.

    Bar colours indicate clinical performance tier:
      Green  (>=50%): Acceptable for a deployment context.
      Orange (>=30%): Borderline - would need clinical oversight.
      Red    (<30%):  Poor - would require model improvement before clinical use.
    """
    range_names     = list(per_range_accuracy.keys())
    accuracy_values = list(per_range_accuracy.values())

    # Colour-code each bar by its clinical performance tier
    bar_colours = [
        "#4CAF50" if acc >= 0.5 else "#FF9800" if acc >= 0.3 else "#F44336"
        for acc in accuracy_values
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        range_names, accuracy_values,
        color=bar_colours, alpha=0.85,
        edgecolor="white", width=0.6,
    )
    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5, label="50% line")
    ax.axhline(
        0.8, color="green", linestyle="--",
        alpha=0.5, label="80% clinical benchmark",
    )
    ax.set_title(
        "Per-Range Accuracy - ResNet-50 + Ordinal Loss + TTA",
        fontweight="bold", fontsize=13,
    )
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Label each bar with its numeric accuracy value for easy reading
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, "per_range_accuracy.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  OK Per-range accuracy -> {output_path}")


# ==================================================================
# SUMMARY TABLE
# ==================================================================

def print_summary_table(resnet_results: dict):
    """
    Print a formatted comparison table of all available model results.

    Loads LLM baseline results (Claude, GPT-4o) from JSON if they exist,
    allowing a side-by-side comparison for the dissertation results section.
    """
    all_results = {"resnet50": resnet_results}

    # Load LLM comparison results if they have already been generated
    for file_tag, model_name in [("claude", "Claude (Anthropic)"),
                                  ("gpt4o", "GPT-4o (OpenAI)")]:
        result_path = os.path.join(RESULTS_DIR, f"{file_tag}_results.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                data = json.load(f)
                data["model"] = model_name
                all_results[file_tag] = data

    print("\n" + "=" * 80)
    print("  RESULTS TABLE - copy into dissertation")
    print("=" * 80)
    print(f"  {'Model':<35} {'Exact':>8} {'Clin.':>8} "
          f"{'Dangerous':>11} {'Macro F1':>10}")
    print("  " + "─" * 78)

    for result in all_results.values():
        macro_avg = result.get("classification_report", {}).get("macro avg", {})
        print(
            f"  {result['model']:<35} "
            f"{result.get('test_accuracy', 0):>7.3f}  "
            f"{result.get('clinically_acceptable_accuracy_+-1_range', 0):>7.3f}  "
            f"{result.get('dangerous_predictions_+-2_ranges', 0):>10.3f}  "
            f"{macro_avg.get('f1-score', 0):>9.3f}"
        )

    print("  " + "─" * 78)
    print("  Clin. = ±1 range (±20g) - Özkaya et al. (2026)")
    print("  Dangerous = ±2+ ranges (±40g) - clinically unsafe")
    print("=" * 80)


# ==================================================================
# ENTRY POINT
# ==================================================================

def evaluate():
    """
    Run the full evaluation pipeline on the held-out test set:
      1. Load the best checkpoint saved during training.
      2. Run TTA inference over all test images.
      3. Compute and print clinical metrics.
      4. Save confusion matrix and per-range accuracy figures.
      5. Print the results comparison table for the dissertation.
      6. Save results to JSON for use in LLM comparison scripts.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load the held-out test set - Food-101 TEST split, unseen during training
    eval_dir = str(Path(DATASET_DIR) / "final_eval")
    test_dataset = CarbRangeDataset(eval_dir, VAL_TF, split=None)
    print(f"\n  Test set: {len(test_dataset):,} images")

    print("\n  Loading ResNet-50...")
    model = load_model()

    print(f"  Running inference with TTA (x8 augmentations per image)...")
    true_labels, pred_labels, _ = run_inference_tta(
        model, test_dataset.samples, n_aug=8
    )

    results = compute_metrics(true_labels, pred_labels)
    plot_confusion_matrix(true_labels, pred_labels)
    plot_per_range_accuracy(results["per_range_accuracy"])
    print_summary_table(results)

    output_path = os.path.join(RESULTS_DIR, "resnet50_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  OK Results saved -> {output_path}")
    print(f"  OK All figures saved to {RESULTS_DIR}/")
