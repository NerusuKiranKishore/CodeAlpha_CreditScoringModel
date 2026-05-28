"""
feature_engineering.py
-----------------------
Creates new meaningful features from raw financial data.
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features to improve model performance.

    New Features
    ------------
    - TotalLatePayments     : sum of all late payment columns
    - DebtToIncome          : DebtRatio * MonthlyIncome (approx monthly debt)
    - IncomePerDependent    : MonthlyIncome / (NumberOfDependents + 1)
    - CreditLineUtilBucket  : binned RevolvingUtilization (Low/Med/High/Very High)
    - AgeGroup              : binned age (Young/Adult/Middle/Senior)
    - HasLatePayment        : binary flag — any late payment in history
    """
    df = df.copy()

    # ── Late payment aggregation ───────────────────────────────────────────────
    late_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    available_late = [c for c in late_cols if c in df.columns]
    if available_late:
        df["TotalLatePayments"] = df[available_late].sum(axis=1)
        df["HasLatePayment"]    = (df["TotalLatePayments"] > 0).astype(int)
        print(f"[FEATURE] Created TotalLatePayments, HasLatePayment")

    # ── Debt-to-income proxy ───────────────────────────────────────────────────
    if "DebtRatio" in df.columns and "MonthlyIncome" in df.columns:
        df["DebtToIncome"] = df["DebtRatio"] * df["MonthlyIncome"]
        df["DebtToIncome"] = df["DebtToIncome"].clip(upper=df["DebtToIncome"].quantile(0.99))
        print("[FEATURE] Created DebtToIncome")

    # ── Income per dependent ───────────────────────────────────────────────────
    if "MonthlyIncome" in df.columns and "NumberOfDependents" in df.columns:
        df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
        print("[FEATURE] Created IncomePerDependent")

    # ── Revolving utilization bucket ──────────────────────────────────────────
    if "RevolvingUtilizationOfUnsecuredLines" in df.columns:
        df["CreditLineUtilBucket"] = pd.cut(
            df["RevolvingUtilizationOfUnsecuredLines"],
            bins=[-0.01, 0.3, 0.6, 0.9, 1.01],
            labels=[0, 1, 2, 3],  # 0=Low, 1=Med, 2=High, 3=Very High
        ).astype(int)
        print("[FEATURE] Created CreditLineUtilBucket")

    # ── Age group ─────────────────────────────────────────────────────────────
    if "age" in df.columns:
        df["AgeGroup"] = pd.cut(
            df["age"],
            bins=[0, 25, 40, 60, 120],
            labels=[0, 1, 2, 3],  # Young / Adult / Middle / Senior
        ).astype(int)
        print("[FEATURE] Created AgeGroup")

    print(f"[FEATURE] Total features after engineering: {df.shape[1]}\n")
    return df