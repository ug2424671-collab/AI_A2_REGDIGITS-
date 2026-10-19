"""End-to-end orchestration: data -> validation -> models -> artifacts."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .clustering import run_clustering
from .config import CLS_TARGET, FEATURES, REG_TARGET
from .data_generation import generate_dataset, write_dataset, write_fingerprint
from .modeling import run_classification, run_regression
from .plots import plot_clusters, plot_confusion, plot_residuals
from .preprocessing import clean_data, split_data
from .validation import check_no_leakage, parse_identity, validate_schema


def _dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def run_pipeline(student_id: str, seed: int, root: Path) -> dict:
    """Regenerate data and every model/artifact under `root`. Returns a summary dict."""
    root = Path(root)
    ident = parse_identity(student_id, seed)
    code = ident["verification_code"]
    for sub in ("data", "models", "artifacts/plots"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    art, mdl = root / "artifacts", root / "models"

    # 1. Data generation + fingerprint
    csv_path = root / "data" / f"AI_A2_{ident['reg_digits']}.csv"
    write_dataset(generate_dataset(ident["seed"]), csv_path)
    fingerprint = write_fingerprint(csv_path, art / "data_fingerprint.txt", code)

    # 2. Validation + cleaning
    raw = pd.read_csv(csv_path)
    raw_report = validate_schema(raw)
    df, clean_report = clean_data(raw)
    check_no_leakage(FEATURES)
    _dump({"verification_code": code, "raw": raw_report, "cleaning": clean_report,
           "leakage_check": "passed: yield_tons, yield_per_ha and high_risk are not features",
           "features": FEATURES}, art / "data_validation_report.json")

    # 3. Split (train-only fitting happens inside the pipelines)
    split = split_data(df, ident["seed"])

    # 4. Regression, classification, clustering
    reg_report, reg_model, _ = run_regression(split, ident["seed"])
    cls_report, cls_model, _ = run_classification(split, ident["seed"])
    clu_report, cprep, kmeans, Ztr, ltr, lte = run_clustering(split, df, ident["seed"])
    Zte = cprep.transform(split["X_test"])

    for rep in (reg_report, cls_report, clu_report):
        rep["verification_code"] = code
    _dump(reg_report, art / "regression_metrics.json")
    _dump(cls_report, art / "classification_metrics.json")
    _dump(clu_report, art / "clustering_report.json")

    # 5. Test predictions
    Xte = split["X_test"]
    pred_y = reg_model.predict(Xte)
    prob = cls_model.predict_proba(Xte)[:, 1]
    preds = pd.DataFrame({
        "row_id": Xte.index,
        "actual_yield_tons": split["y_reg_test"].values,
        "predicted_yield_tons": pred_y,
        "yield_error": split["y_reg_test"].values - pred_y,
        "yield_abs_error": np.abs(split["y_reg_test"].values - pred_y),
        "actual_high_risk": split["y_cls_test"].values,
        "predicted_high_risk": (prob >= 0.5).astype(int),
        "high_risk_probability": prob,
        "risk_error": (split["y_cls_test"].values != (prob >= 0.5).astype(int)).astype(int),
        "cluster": lte,
    })
    preds.to_csv(art / "test_predictions.csv", index=False, float_format="%.4f",
                 lineterminator="\n")

    # 6. Plots
    plot_residuals(split["y_reg_test"], pred_y, art / "plots" / "residuals.png", code)
    best_cls = cls_report["selected_model"]
    plot_confusion(cls_report["candidates"][best_cls]["confusion_matrix"],
                   art / "plots" / "confusion_matrix.png", code, best_cls)
    plot_clusters(Ztr, ltr, Zte, lte, art / "plots" / "clusters.png", code)

    # 7. Save models + info used by predict.py and app.py
    joblib.dump(reg_model, mdl / "regression_model.joblib")
    joblib.dump(cls_model, mdl / "classification_model.joblib")
    joblib.dump(cprep, mdl / "cluster_preprocessor.joblib")
    joblib.dump(kmeans, mdl / "kmeans.joblib")
    info = {
        "verification_code": code, "student_id": student_id, "seed": ident["seed"],
        "dataset_rows_raw": raw_report["rows"], "dataset_rows_clean": len(df),
        "dataset_sha256": fingerprint, "features": FEATURES,
        "regression_model": reg_report["selected_model"],
        "regression_test_metrics": reg_report["candidates"][reg_report["selected_model"]],
        "classification_model": best_cls,
        "classification_test_metrics": cls_report["candidates"][best_cls],
        "clustering_k": clu_report["selected_k"],
        "cluster_descriptions": {k: v["description"]
                                 for k, v in clu_report["profiles"].items()},
        "synthetic_data": True,
        "intended_use": "Classroom decision-support prototype; human review required.",
    }
    _dump(info, mdl / "model_info.json")

    return {"identity": ident, "fingerprint": fingerprint, "csv": csv_path.name,
            "raw_rows": raw_report["rows"], "clean_rows": len(df),
            "missing": raw_report["missing_values"],
            "duplicates": raw_report["duplicate_rows"],
            "regression": reg_report, "classification": cls_report, "clustering": clu_report}
