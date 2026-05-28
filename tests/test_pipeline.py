"""
tests/test_pipeline.py
-----------------------
Unit tests for each stage of the pipeline.
Run with: python -m pytest tests/ -v
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing       import handle_missing_values, cap_outliers, scale_features
from feature_engineering import engineer_features


# ── Fixture — minimal synthetic dataframe ─────────────────────────────────────
@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "SeriousDlqin2yrs"                      : np.random.randint(0, 2, n),
        "RevolvingUtilizationOfUnsecuredLines"   : np.random.uniform(0, 1.5, n),
        "age"                                    : np.random.randint(18, 80, n),
        "NumberOfTime30-59DaysPastDueNotWorse"   : np.random.randint(0, 5, n),
        "DebtRatio"                              : np.random.uniform(0, 1.5, n),
        "MonthlyIncome"                          : np.where(
                                                        np.random.rand(n) < 0.1,
                                                        np.nan,
                                                        np.random.randint(1000, 10000, n)
                                                    ),
        "NumberOfOpenCreditLinesAndLoans"        : np.random.randint(0, 20, n),
        "NumberOfTimes90DaysLate"                : np.random.randint(0, 5, n),
        "NumberRealEstateLoansOrLines"           : np.random.randint(0, 5, n),
        "NumberOfTime60-89DaysPastDueNotWorse"   : np.random.randint(0, 5, n),
        "NumberOfDependents"                     : np.where(
                                                        np.random.rand(n) < 0.05,
                                                        np.nan,
                                                        np.random.randint(0, 5, n)
                                                    ),
    })


# ── Preprocessing tests ───────────────────────────────────────────────────────

def test_no_missing_after_imputation(sample_df):
    df_clean = handle_missing_values(sample_df)
    assert df_clean["MonthlyIncome"].isna().sum() == 0
    assert df_clean["NumberOfDependents"].isna().sum() == 0


def test_outlier_capping(sample_df):
    df_clean = handle_missing_values(sample_df)
    df_capped = cap_outliers(df_clean)
    assert df_capped["RevolvingUtilizationOfUnsecuredLines"].max() <= 1.0
    assert df_capped["DebtRatio"].max() <= 1.0
    assert df_capped["age"].min() >= 18
    assert df_capped["age"].max() <= 100


def test_scaling_shape(sample_df):
    df_clean  = handle_missing_values(sample_df)
    X         = df_clean.drop(columns=["SeriousDlqin2yrs"])
    half      = len(X) // 2
    X_tr, X_te, _ = X[:half], X[half:], None
    X_tr_sc, X_te_sc, scaler = scale_features(X_tr, X_te)
    assert X_tr_sc.shape == X_tr.shape
    assert X_te_sc.shape == X_te.shape


# ── Feature engineering tests ─────────────────────────────────────────────────

def test_new_features_exist(sample_df):
    df_clean = handle_missing_values(sample_df)
    df_feat  = engineer_features(df_clean)
    expected = [
        "TotalLatePayments", "HasLatePayment",
        "DebtToIncome", "IncomePerDependent",
        "CreditLineUtilBucket", "AgeGroup",
    ]
    for col in expected:
        assert col in df_feat.columns, f"Missing feature: {col}"


def test_has_late_payment_binary(sample_df):
    df_clean = handle_missing_values(sample_df)
    df_feat  = engineer_features(df_clean)
    assert set(df_feat["HasLatePayment"].unique()).issubset({0, 1})


def test_feature_count_increases(sample_df):
    df_clean = handle_missing_values(sample_df)
    before   = df_clean.shape[1]
    df_feat  = engineer_features(df_clean)
    assert df_feat.shape[1] > before