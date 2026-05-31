"""
evaluate.py
-----------
Generates evaluation metrics and saves visualisation plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)


# ── Output directory for figures ──────────────────────────────────────────────
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Individual model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Print classification report and save confusion matrix + ROC curve.

    Returns
    -------
    metrics : dict with accuracy, roc_auc, avg_precision
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Text report ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {model_name} — Evaluation Report")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, target_names=["Good Credit", "Bad Credit"]))

    roc_auc   = roc_auc_score(y_test, y_prob)
    avg_prec  = average_precision_score(y_test, y_prob)
    print(f"  ROC-AUC Score        : {roc_auc:.4f}")
    print(f"  Avg Precision Score  : {avg_prec:.4f}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    _plot_confusion_matrix(y_test, y_pred, model_name)

    # ── ROC curve ─────────────────────────────────────────────────────────────
    _plot_roc_curve(y_test, y_prob, model_name, roc_auc)

    return {
        "model_name"    : model_name,
        "roc_auc"       : roc_auc,
        "avg_precision" : avg_prec,
    }


def _plot_confusion_matrix(y_test, y_pred, model_name: str):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Good Credit", "Bad Credit"],
        yticklabels=["Good Credit", "Bad Credit"],
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    safe = model_name.replace(" ", "_").lower()
    path = os.path.join(FIGURES_DIR, f"confusion_matrix_{safe}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [EVAL] Confusion matrix saved → {path}")


def _plot_roc_curve(y_test, y_prob, model_name: str, roc_auc: float):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    safe = model_name.replace(" ", "_").lower()
    path = os.path.join(FIGURES_DIR, f"roc_curve_{safe}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [EVAL] ROC curve saved        → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Comparison across all models
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(results: dict):
    """
    Bar chart comparing ROC-AUC across all trained models.
    """
    names   = list(results.keys())
    scores  = [results[n]["roc_auc"] for n in names]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, scores, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    plt.ylim(0.5, 1.0)
    plt.ylabel("ROC-AUC Score")
    plt.title("Model Comparison — ROC-AUC")
    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{score:.4f}",
            ha="center", va="bottom", fontsize=11,
        )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n[EVAL] Model comparison chart saved → {path}")


def plot_feature_importance(model, feature_names: list, model_name: str, top_n: int = 15):
    """
    Plot feature importances for tree-based models.
    """
    if not hasattr(model, "feature_importances_"):
        print(f"[EVAL] {model_name} has no feature_importances_ — skipping.")
        return

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=importances[indices],
        y=[feature_names[i] for i in indices],
        palette="viridis",
    )
    plt.title(f"Top {top_n} Feature Importances — {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    safe = model_name.replace(" ", "_").lower()
    path = os.path.join(FIGURES_DIR, f"feature_importance_{safe}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[EVAL] Feature importance plot saved → {path}")


def evaluate_all(results: dict, X_test, y_test, feature_names: list):
    """
    Evaluate every model in `results` dict and generate comparison chart.
    """
    for name, info in results.items():
        evaluate_model(info["model"], X_test, y_test, name)
        plot_feature_importance(info["model"], feature_names, name)

    plot_model_comparison(results)