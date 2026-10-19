"""Load saved artifacts and score one validated farm record."""
import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from .config import DISCLAIMER, FEATURES, PROJECT_ROOT
from .schemas import FarmRecord


class ModelsNotFoundError(RuntimeError):
    """Raised when saved models are missing (run_all.py has not been executed)."""


def get_root() -> Path:
    """Project root; AI_A2_ROOT overrides it (used by tests)."""
    return Path(os.environ.get("AI_A2_ROOT", PROJECT_ROOT))


@lru_cache(maxsize=4)
def load_bundle(root: str) -> dict:
    models = Path(root) / "models"
    needed = ["regression_model.joblib", "classification_model.joblib",
              "cluster_preprocessor.joblib", "kmeans.joblib", "model_info.json"]
    missing = [n for n in needed if not (models / n).exists()]
    if missing:
        raise ModelsNotFoundError(
            f"Missing model files {missing} in {models}. Run: python run_all.py "
            "--student-id <id> --seed <LLLL>")
    return {
        "reg": joblib.load(models / "regression_model.joblib"),
        "cls": joblib.load(models / "classification_model.joblib"),
        "cluster_prep": joblib.load(models / "cluster_preprocessor.joblib"),
        "kmeans": joblib.load(models / "kmeans.joblib"),
        "info": json.loads((models / "model_info.json").read_text(encoding="utf-8")),
    }


def get_model_info() -> dict:
    return load_bundle(str(get_root()))["info"]


def predict_validated(record: FarmRecord) -> dict:
    """Score a record that already passed FarmRecord validation."""
    b = load_bundle(str(get_root()))
    X = pd.DataFrame([record.model_dump()])[FEATURES]
    prob = float(b["cls"].predict_proba(X)[0, 1])
    risk = int(prob >= 0.5)
    cluster = int(b["kmeans"].predict(b["cluster_prep"].transform(X))[0])
    profile = b["info"]["cluster_descriptions"][str(cluster)]
    return {
        "yield_tons_estimate": round(float(b["reg"].predict(X)[0]), 4),
        "high_risk_prediction": risk,
        "high_risk_label": "high risk" if risk else "not high risk",
        "high_risk_probability": round(prob, 4),
        "cluster": cluster,
        "cluster_profile": profile,
        "verification_code": b["info"]["verification_code"],
        "disclaimer": DISCLAIMER,
    }


def predict_record(record: dict) -> dict:
    """Validate a raw dict (raises pydantic.ValidationError) and score it."""
    return predict_validated(FarmRecord.model_validate(record))
