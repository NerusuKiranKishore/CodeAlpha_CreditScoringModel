"""
train.py
--------
Trains multiple classifiers and saves the best model.
"""

import os
import joblib
import numpy as np

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from xgboost                 import XGBClassifier
from imblearn.over_sampling  import SMOTE
from sklearn.metrics         import roc_auc_score


# ── Where to save trained models ──────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def get_models() -> dict:
    """
    Return a dictionary of model name → estimator instance.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            random_state=42,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }


def apply_smote(X_train, y_train, random_state: int = 42):
    """
    Handle class imbalance using SMOTE oversampling on training data.
    """
    print(f"[TRAIN] Before SMOTE — class counts: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"[TRAIN] After  SMOTE — class counts: {dict(zip(*np.unique(y_res, return_counts=True)))}\n")
    return X_res, y_res


def train_all_models(
    X_train,
    y_train,
    X_test,
    y_test,
    use_smote: bool = True,
) -> dict:
    """
    Train every model, evaluate ROC-AUC on test set, save all.

    Returns
    -------
    results : dict  {model_name: {"model": fitted_model, "roc_auc": float}}
    """
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)

    models  = get_models()
    results = {}

    for name, model in models.items():
        print(f"[TRAIN] Training {name} ...")
        model.fit(X_train, y_train)

        y_prob  = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"[TRAIN] {name} → ROC-AUC: {roc_auc:.4f}")

        # Save model
        safe_name  = name.replace(" ", "_").lower()
        model_path = os.path.join(MODELS_DIR, f"{safe_name}.pkl")
        joblib.dump(model, model_path)
        print(f"[TRAIN] Saved → {model_path}\n")

        results[name] = {"model": model, "roc_auc": roc_auc}

    return results


def save_best_model(results: dict, scaler) -> str:
    """
    Identify the best model by ROC-AUC and save it as 'best_model.pkl'.
    Also save the scaler.

    Returns
    -------
    best_name : str
    """
    best_name  = max(results, key=lambda n: results[n]["roc_auc"])
    best_model = results[best_name]["model"]

    best_path   = os.path.join(MODELS_DIR, "best_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

    joblib.dump(best_model, best_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n[TRAIN] ✅ Best model: {best_name}  (ROC-AUC = {results[best_name]['roc_auc']:.4f})")
    print(f"[TRAIN] Saved best model → {best_path}")
    print(f"[TRAIN] Saved scaler     → {scaler_path}\n")

    return best_name