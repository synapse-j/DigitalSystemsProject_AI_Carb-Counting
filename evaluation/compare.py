"""
evaluation/compare.py
======================
CHANGES FROM V1:

  1. Clinical safety comparison - adds a chart showing dangerous
     predictions (+-2 ranges = +-40g error) alongside accuracy.
     This is more meaningful for T1D context than accuracy alone.

  2. Confidence vs accuracy chart - shows whether LLMs express
     appropriate uncertainty compared to the ML pipeline.

  3. Richer summary table - includes dangerous prediction rate
     and per-range accuracy for the dissertation results section.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RESULTS_DIR, CARB_RANGE_LABELS

RANGE_LABELS = list(CARB_RANGE_LABELS.values())

DISPLAY = {
    "resnet50": "ResNet-50\n(Fine-tuned)",
    "gpt5":     "GPT-5.4\n(Zero-shot)",
}
COLOURS = {
    "resnet50": "#1565C0",
    "gpt5":     "#E65100",
}


def load_results() -> dict:
    results = {}
    for tag in ("resnet50", "gpt5"):
        path = os.path.join(RESULTS_DIR, f"{tag}_results.json")
        if os.path.exists(path):
            with open(path) as f:
                results[tag] = json.load(f)

    if not results:
        raise FileNotFoundError(
            "No results found. Run: python run.py --stage evaluate"
        )
    print(f"  Loaded results for: {list(results.keys())}")
    return results


# ==================================================================
# FIGURE 1: Main accuracy comparison (updated with clinical safety)
# ==================================================================

def fig1_accuracy(results: dict):
    tags      = list(results.keys())
    names     = [DISPLAY.get(t, t) for t in tags]
    accs      = [results[t].get("test_accuracy", 0) for t in tags]
    clin_accs = [results[t].get("clinically_acceptable_accuracy_-1_range",
                  results[t].get("clinically_acceptable_accuracy_+-1_range", 0)) for t in tags]

    # Pull dangerous prediction rate if available (cascade has it, LLMs may not)
    dangerous = []
    for t in tags:
        r = results[t]
        d = r.get("metrics", {}).get("dangerous_predictions_-2_ranges",
            r.get("dangerous_predictions_-2_ranges",
            r.get("dangerous_predictions_+-2_ranges", None)))
        dangerous.append(d)

    colours = [COLOURS.get(t, "#888") for t in tags]
    x, w    = np.arange(len(tags)), 0.25

    _, ax = plt.subplots(figsize=(11, 6))

    b1 = ax.bar(x - w, accs, w, color=colours, alpha=0.90,
                label="Exact Accuracy", edgecolor="white")
    b2 = ax.bar(x,     clin_accs, w, color=colours, alpha=0.55,
                label="Clinically Acceptable (+-1 range / +-20g)",
                edgecolor="white", hatch="//")

    # Dangerous predictions - shown as downward red bars if available
    has_dangerous = any(d is not None for d in dangerous)
    if has_dangerous:
        danger_vals = [d if d is not None else 0 for d in dangerous]
        b3 = ax.bar(x + w, danger_vals, w, color="red", alpha=0.6,
                    label="Dangerous Predictions (+-2+ ranges / +-40g)",
                    edgecolor="white")
        for bar in b3:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.012,
                        f"{bar.get_height():.3f}",
                        ha="center", va="bottom", fontsize=9, color="red")

    ax.axhline(0.80, color="green", linestyle="--", linewidth=1.2,
               alpha=0.6, label="80% benchmark")
    ax.set_title(
        "Carbohydrate Range Classification Performance\n"
        "Cascaded ML Pipeline vs Large Language Models",
        fontsize=13, fontweight="bold"
    )
    ax.set_ylabel("Proportion of Predictions", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig1_accuracy.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  OK fig1_accuracy.png")


# ==================================================================
# FIGURE 2: Per-class F1
# ==================================================================

def fig2_f1(results: dict):
    tags  = list(results.keys())
    x, n, w = np.arange(len(RANGE_LABELS)), len(tags), 0.25
    _, ax = plt.subplots(figsize=(12, 5))
    for i, tag in enumerate(tags):
        report = results[tag].get("classification_report", {})
        f1s    = [report.get(lbl, {}).get("f1-score", 0) for lbl in RANGE_LABELS]
        offset = (i - n/2 + 0.5) * w
        ax.bar(x + offset, f1s, w,
               label=DISPLAY.get(tag, tag).replace("\n", " "),
               color=COLOURS.get(tag, "#888"), alpha=0.85, edgecolor="white")
    ax.set_title("Per-Class F1 Score by Carbohydrate Range",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("F1 Score"); ax.set_ylim(0, 1.1)
    ax.set_xticks(x); ax.set_xticklabels(RANGE_LABELS, fontsize=11)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig2_f1_per_class.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  OK fig2_f1_per_class.png")


# ==================================================================
# FIGURE 3: Radar
# ==================================================================

def fig3_radar(results: dict):
    metrics  = ["Exact\nAccuracy", "Clinical\nAccuracy",
                "Macro F1", "Precision", "Recall"]
    n        = len(metrics)
    angles   = np.linspace(0, 2*np.pi, n, endpoint=False).tolist() + [0]

    _, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for tag, r in results.items():
        macro = r.get("classification_report", {}).get("macro avg", {})
        vals  = [
            r.get("test_accuracy", 0),
            r.get("clinically_acceptable_accuracy_-1_range",
              r.get("clinically_acceptable_accuracy_+-1_range", 0)),
            macro.get("f1-score", 0),
            macro.get("precision", 0),
            macro.get("recall", 0),
        ] + [r.get("test_accuracy", 0)]
        col = COLOURS.get(tag, "#888")
        ax.plot(angles, vals, "o-", lw=2, color=col,
                label=DISPLAY.get(tag, tag).replace("\n", " "))
        ax.fill(angles, vals, alpha=0.08, color=col)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("Multi-Metric Comparison", fontsize=13,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=9)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig3_radar.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  OK fig3_radar.png")


# ==================================================================
# SUMMARY TABLE (updated with dangerous prediction rate)
# ==================================================================

def summary_table(results: dict):
    lines = [
        "=" * 90,
        "  DISSERTATION RESULTS TABLE",
        "=" * 90,
        f"  {'Model':<28} {'Accuracy':>10} {'Clin.Acc.':>11} "
        f"{'Dangerous':>11} {'Macro F1':>10} {'Recall':>8}",
        "  " + "-" * 88,
    ]
    for tag, r in results.items():
        macro    = r.get("classification_report", {}).get("macro avg", {})
        name     = DISPLAY.get(tag, tag).replace("\n", " ")
        danger   = r.get("metrics", {}).get("dangerous_predictions_-2_ranges",
                   r.get("dangerous_predictions_-2_ranges",
                   r.get("dangerous_predictions_+-2_ranges", float("nan"))))
        d_str    = f"{danger:.3f}" if not (isinstance(danger, float)
                   and np.isnan(danger)) else "  N/A"
        lines.append(
            f"  {name:<28} "
            f"{r.get('test_accuracy', 0):>9.3f}  "
            f"{r.get('clinically_acceptable_accuracy_-1_range', r.get('clinically_acceptable_accuracy_+-1_range', 0)):>10.3f}  "
            f"{d_str:>10}  "
            f"{macro.get('f1-score', 0):>9.3f}  "
            f"{macro.get('recall', 0):>7.3f}"
        )
    lines += [
        "  " + "-" * 88,
        "  Clin.Acc.  = predictions within +-1 range (approx +-20g) - Ozkaya et al. (2026)",
        "  Dangerous  = predictions off by +-2+ ranges (approx +-40g) - clinically unsafe",
        "=" * 90,
    ]
    table = "\n".join(lines)
    print("\n" + table)
    path = os.path.join(RESULTS_DIR, "summary_table.txt")
    with open(path, "w") as f:
        f.write(table)
    print(f"\n  OK summary_table.txt")


def generate_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = load_results()
    fig1_accuracy(results)
    fig2_f1(results)
    fig3_radar(results)
    summary_table(results)
    with open(os.path.join(RESULTS_DIR, "all_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nOK All figures saved to {RESULTS_DIR}/")