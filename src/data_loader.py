"""
data_loader.py
--------------
Handles loading the dataset and splitting into train/test sets.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ── Column name used as the target label ──────────────────────────────────────
TARGET_COLUMN = "SeriousDlqin2yrs"

# ── Default path to raw data ──────────────────────────────────────────────────
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "credit_data.csv"
)


def load_data(filepath: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Load the CSV dataset into a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at: {filepath}"
            "\nPlease download 'cs-training.csv' from Kaggle:"
            "\nhttps://www.kaggle.com/c/GiveMeSomeCredit"
            "\nRename it to 'credit_data.csv' and place it in data/raw/"
        )

    df = pd.read_csv(filepath, index_col=0)
    print(f"[INFO] Dataset loaded — shape: {df.shape}")
    print(f"[INFO] Target distribution:\n{df[TARGET_COLUMN].value_counts()}\n")
    return df


def split_data(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split DataFrame into train and test sets.

    Parameters
    ----------
    df           : cleaned DataFrame
    target       : name of the target column
    test_size    : fraction for test split (default 0.20)
    random_state : seed for reproducibility

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[INFO] Train size : {X_train.shape[0]} samples")
    print(f"[INFO] Test  size : {X_test.shape[0]} samples\n")
    return X_train, X_test, y_train, y_test