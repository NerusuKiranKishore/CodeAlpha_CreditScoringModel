"""
preprocessing.py
----------------
Handles missing values, outlier capping, and feature scaling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ── Columns with known missing values in the GiveMeSomeCredit dataset ─────────
MEDIAN_IMPUTE_COLS = ["MonthlyIncome"]
ZERO_IMPUTE_COLS   = ["NumberOfDependents"]

# ── Outlier caps (99th percentile — values above are clipped) ──────────────────
OUTLIER_CAPS = {
    "RevolvingUtilizationOfUnsecuredLines": 1.0,
    "DebtRatio": 1.0,
    "age": (18, 100),
}


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values.
    - MonthlyIncome  → median imputation
    - NumberOfDependents → 0 imputation
    """
    df = df.copy()

    for col in MEDIAN_IMPUTE_COLS:
        if col in df.columns:
            median_val = df[col].median()
            missing = df[col].isna().sum()
            df[col].fillna(median_val, inplace=True)
            print(f"[PREPROCESS] {col}: filled {missing} NaNs with median ({median_val:.2f})")

    for col in ZERO_IMPUTE_COLS:
        if col in df.columns:
            missing = df[col].isna().sum()
            df[col].fillna(0, inplace=True)
            print(f"[PREPROCESS] {col}: filled {missing} NaNs with 0")

    return df


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap extreme outliers using predefined thresholds.
    """
    df = df.copy()

    for col, cap in OUTLIER_CAPS.items():
        if col not in df.columns:
            continue
        if isinstance(cap, tuple):
            lo, hi = cap
            df[col] = df[col].clip(lower=lo, upper=hi)
        else:
            df[col] = df[col].clip(upper=cap)
        print(f"[PREPROCESS] {col}: outliers capped at {cap}")

    return df


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """
    Fit StandardScaler on training data and transform both sets.

    Returns
    -------
    X_train_scaled (np.ndarray), X_test_scaled (np.ndarray), scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print("[PREPROCESS] Features scaled with StandardScaler\n")
    return X_train_scaled, X_test_scaled, scaler


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline: missing values → outlier capping.
    """
    df = handle_missing_values(df)
    df = cap_outliers(df)
    return df