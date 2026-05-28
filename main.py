"""
main.py
-------
Full end-to-end pipeline:
  Load → Preprocess → Feature Engineering → Split → Scale → Train → Evaluate
"""

import os
import sys

# Make src importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader          import load_data, split_data, TARGET_COLUMN
from preprocessing        import preprocess, scale_features
from feature_engineering  import engineer_features
from train                import train_all_models, save_best_model
from evaluate             import evaluate_all


def run_pipeline(data_path: str = None):
    print("\n" + "=" * 65)
    print("   💳  Credit Scoring Model — CodeAlpha ML Internship")
    print("=" * 65 + "\n")

    # ── 1. Load ────────────────────────────────────────────────────────────────
    df = load_data(data_path) if data_path else load_data()

    # ── 2. Preprocess ──────────────────────────────────────────────────────────
    print("[STEP 2] Preprocessing ...")
    df = preprocess(df)

    # ── 3. Feature Engineering ─────────────────────────────────────────────────
    print("[STEP 3] Feature Engineering ...")
    df = engineer_features(df)

    # ── 4. Train / Test Split ──────────────────────────────────────────────────
    print("[STEP 4] Splitting data ...")
    X_train, X_test, y_train, y_test = split_data(df, target=TARGET_COLUMN)
    feature_names = X_train.columns.tolist()

    # ── 5. Scale ───────────────────────────────────────────────────────────────
    print("[STEP 5] Scaling features ...")
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    # ── 6. Train ───────────────────────────────────────────────────────────────
    print("[STEP 6] Training models ...")
    results = train_all_models(X_train_sc, y_train, X_test_sc, y_test)

    # ── 7. Evaluate ────────────────────────────────────────────────────────────
    print("[STEP 7] Evaluating models ...")
    evaluate_all(results, X_test_sc, y_test, feature_names)

    # ── 8. Save best model ─────────────────────────────────────────────────────
    print("[STEP 8] Saving best model ...")
    best_name = save_best_model(results, scaler)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("   ✅  Pipeline Complete!")
    print(f"   🏆  Best Model  : {best_name}")
    print(f"   📊  ROC-AUC     : {results[best_name]['roc_auc']:.4f}")
    print("   📁  Plots saved in  : reports/figures/")
    print("   💾  Models saved in : models/")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    # Optionally pass a custom data path as CLI argument
    custom_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(custom_path)