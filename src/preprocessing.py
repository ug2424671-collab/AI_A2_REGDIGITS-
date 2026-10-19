"""Cleaning, train/test split and the shared preprocessing ColumnTransformer."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (BINARY_FEATURES, CATEGORICAL_FEATURES, CLS_TARGET, FEATURES,
                     NUMERIC_FEATURES, REG_TARGET, TEST_SIZE)
from .validation import check_no_leakage


def clean_data(df: pd.DataFrame):
    """Drop exact duplicate rows (keeping the first). Missing values are imputed later,
    inside the pipeline, so imputation statistics come from training rows only."""
    before = len(df)
    cleaned = df.drop_duplicates(keep="first")
    return cleaned, {"rows_before": before, "rows_after": len(cleaned),
                     "duplicates_removed": before - len(cleaned)}


def build_preprocessor() -> ColumnTransformer:
    """Median impute + scale numeric/binary columns; mode impute + one-hot season."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                     sparse_output=False))])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES + BINARY_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def split_data(df: pd.DataFrame, seed: int) -> dict:
    """One fixed, stratified 80/20 split shared by every task (random_state = LLLL).

    The DataFrame index is kept as the row id for test_predictions.csv.
    """
    check_no_leakage(FEATURES)
    train_idx, test_idx = train_test_split(
        df.index, test_size=TEST_SIZE, random_state=seed, stratify=df[CLS_TARGET])
    tr, te = df.loc[train_idx], df.loc[test_idx]
    return {
        "X_train": tr[FEATURES], "X_test": te[FEATURES],
        "y_reg_train": tr[REG_TARGET], "y_reg_test": te[REG_TARGET],
        "y_cls_train": tr[CLS_TARGET], "y_cls_test": te[CLS_TARGET],
    }
