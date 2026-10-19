import json

import pandas as pd
import pytest

from src.config import FEATURES
from src.data_generation import generate_dataset, sha256_of_file, write_dataset
from src.pipeline import run_pipeline
from src.validation import check_no_leakage, parse_identity, validate_schema
from tests.conftest import TEST_ID, TEST_SEED


def test_reproducibility_same_fingerprint_and_metrics(trained_root, tmp_path):
    second = run_pipeline(TEST_ID, TEST_SEED, tmp_path)
    first_fp = (trained_root / "artifacts" / "data_fingerprint.txt").read_text()
    assert second["fingerprint"] in first_fp
    m1 = json.loads((trained_root / "artifacts" / "regression_metrics.json").read_text())
    m2 = json.loads((tmp_path / "artifacts" / "regression_metrics.json").read_text())
    sel = m1["selected_model"]
    assert m1["candidates"][sel]["rmse"] == pytest.approx(m2["candidates"][sel]["rmse"])


def test_generator_is_deterministic(tmp_path):
    a = write_dataset(generate_dataset(TEST_SEED), tmp_path / "a.csv")
    b = write_dataset(generate_dataset(TEST_SEED), tmp_path / "b.csv")
    assert sha256_of_file(a) == sha256_of_file(b)


def test_wrong_seed_rejected():
    with pytest.raises(ValueError, match="invalid"):
        parse_identity(TEST_ID, 1234)
    assert parse_identity(TEST_ID, 4671)["verification_code"] == "AI-A2-4671"


def test_deliberate_data_problems_detected(tmp_path):
    df = generate_dataset(TEST_SEED)
    n_base = 240 + TEST_SEED % 41
    m = round(0.05 * n_base)
    report = validate_schema(df)
    assert report["rows"] == n_base + 3
    assert report["duplicate_rows"] == 3
    assert report["missing_values"] == {"soil_moisture_pct": m, "fertilizer_kg": m}


def test_schema_rejects_missing_column_and_bad_range():
    df = generate_dataset(TEST_SEED)
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(df.drop(columns=["season"]))
    bad = df.copy()
    bad.loc[0, "rainfall_mm"] = 99999
    with pytest.raises(ValueError, match="Range"):
        validate_schema(bad)


def test_leakage_guard():
    check_no_leakage(FEATURES)
    with pytest.raises(ValueError, match="leakage"):
        check_no_leakage(FEATURES + ["yield_per_ha"])


def test_pipeline_writes_required_artifacts(trained_root):
    for rel in ["artifacts/regression_metrics.json", "artifacts/classification_metrics.json",
                "artifacts/clustering_report.json", "artifacts/test_predictions.csv",
                "artifacts/data_fingerprint.txt", "artifacts/plots/residuals.png",
                "artifacts/plots/confusion_matrix.png", "artifacts/plots/clusters.png",
                "models/regression_model.joblib", "models/classification_model.joblib"]:
        assert (trained_root / rel).exists(), rel
    preds = pd.read_csv(trained_root / "artifacts" / "test_predictions.csv")
    assert {"row_id", "actual_yield_tons", "predicted_yield_tons", "cluster"} <= set(preds)
