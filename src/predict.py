"""
predict.py
----------
Load saved best model + scaler and predict on new input data.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib


MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH     = os.path.join(MODELS_DIR, "scaler.pkl")


def load_artifacts():
    """
    Load the saved best model and scaler.
    """
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            "[ERROR] No saved model found. Please run main.py first."
        )
    model  = joblib.load(BEST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[PREDICT] Model and scaler loaded successfully.\n")
    return model, scaler


def predict_single(input_dict: dict) -> dict:
    """
    Predict creditworthiness for a single applicant.

    Parameters
    ----------
    input_dict : dict of feature_name → value  (raw, unscaled)

    Returns
    -------
    dict with 'label' (Good/Bad Credit) and 'probability'
    """
    model, scaler = load_artifacts()

    df     = pd.DataFrame([input_dict])
    scaled = scaler.transform(df)
    prob   = model.predict_proba(scaled)[0][1]
    label  = "Bad Credit ⚠️" if prob >= 0.5 else "Good Credit ✅"

    print(f"  Prediction  : {label}")
    print(f"  Probability : {prob:.4f} (probability of default)\n")
    return {"label": label, "probability": round(float(prob), 4)}


def predict_batch(filepath: str) -> pd.DataFrame:
    """
    Predict creditworthiness for a CSV file of applicants.

    Parameters
    ----------
    filepath : path to CSV (same columns as training data, no target)

    Returns
    -------
    DataFrame with added 'prediction' and 'default_probability' columns
    """
    model, scaler = load_artifacts()

    df     = pd.read_csv(filepath)
    scaled = scaler.transform(df)
    probs  = model.predict_proba(scaled)[:, 1]
    preds  = ["Bad Credit" if p >= 0.5 else "Good Credit" for p in probs]

    df["prediction"]          = preds
    df["default_probability"] = probs.round(4)

    out_path = filepath.replace(".csv", "_predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"[PREDICT] Batch predictions saved → {out_path}")
    return df


# ── Demo when run directly ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example applicant — edit these values as needed
    sample_applicant = {
        "RevolvingUtilizationOfUnsecuredLines" : 0.45,
        "age"                                  : 35,
        "NumberOfTime30-59DaysPastDueNotWorse" : 0,
        "DebtRatio"                            : 0.30,
        "MonthlyIncome"                        : 5000,
        "NumberOfOpenCreditLinesAndLoans"      : 8,
        "NumberOfTimes90DaysLate"              : 0,
        "NumberRealEstateLoansOrLines"         : 1,
        "NumberOfTime60-89DaysPastDueNotWorse" : 0,
        "NumberOfDependents"                   : 2,
        # Engineered features
        "TotalLatePayments"                    : 0,
        "HasLatePayment"                       : 0,
        "DebtToIncome"                         : 0.30 * 5000,
        "IncomePerDependent"                   : 5000 / 3,
        "CreditLineUtilBucket"                 : 1,
        "AgeGroup"                             : 1,
    }

    print("=" * 50)
    print("  Credit Scoring — Single Applicant Prediction")
    print("=" * 50)
    result = predict_single(sample_applicant)