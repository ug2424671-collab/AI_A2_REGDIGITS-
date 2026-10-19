"""Identity, schema, range, missing-value, duplicate and leakage checks."""
import re

import pandas as pd

from .config import COLUMNS, FORBIDDEN_FEATURES, RANGES, SEASONS


def parse_identity(student_id: str, seed: int) -> dict:
    """Derive registration digits and the AI-A2-LLLL code; reject a seed that is not LLLL."""
    local = student_id.split("@")[0]
    digits = re.sub(r"\D", "", local)
    if len(digits) < 4:
        raise ValueError(
            f"Student id '{student_id}' must contain at least four digits (registration number)."
        )
    last4 = digits[-4:]
    if int(seed) != int(last4):
        raise ValueError(
            f"Seed {seed} is invalid. It must equal the last four registration digits "
            f"({last4} -> {int(last4)})."
        )
    return {
        "student_id": student_id,
        "seed": int(seed),
        "reg_digits": digits,
        "verification_code": f"AI-A2-{last4}",
    }


def validate_schema(df: pd.DataFrame) -> dict:
    """Check columns and ranges; return a report with missing-value and duplicate evidence.

    Raises ValueError for missing columns or out-of-range values (NaN is tolerated here
    because missing values are handled later by imputation).
    """
    missing_cols = [c for c in COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    violations = {}
    for col, (lo, hi) in RANGES.items():
        vals = df[col].dropna()
        n_bad = int(((vals < lo) | (vals > hi)).sum())
        if n_bad:
            violations[col] = n_bad
    bad_season = int((~df["season"].isin(SEASONS)).sum())
    if bad_season:
        violations["season"] = bad_season
    if violations:
        raise ValueError(f"Range/category violations found: {violations}")

    return {
        "rows": int(len(df)),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items() if v},
        "duplicate_rows": int(df.duplicated().sum()),
        "range_violations": 0,
    }


def check_no_leakage(feature_columns) -> None:
    """Raise if a target-derived column is used as a model input."""
    leaked = FORBIDDEN_FEATURES.intersection(feature_columns)
    if leaked:
        raise ValueError(f"Target leakage: {sorted(leaked)} must not be model inputs.")
