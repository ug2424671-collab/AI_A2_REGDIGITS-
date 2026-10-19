"""Regression and classification: baselines, candidates, CV selection, test metrics."""
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .config import CV_FOLDS
from .preprocessing import build_preprocessor


def _pipe(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


def run_regression(split: dict, seed: int):
    """Compare mean baseline with Ridge, RandomForest and GradientBoosting."""
    models = {
        "mean_baseline": _pipe(DummyRegressor(strategy="mean")),
        "ridge": _pipe(Ridge(alpha=1.0)),
        "random_forest": _pipe(RandomForestRegressor(n_estimators=300, random_state=seed)),
        "gradient_boosting": _pipe(GradientBoostingRegressor(random_state=seed)),
    }
    Xtr, ytr = split["X_train"], split["y_reg_train"]
    Xte, yte = split["X_test"], split["y_reg_test"]
    cv = KFold(CV_FOLDS, shuffle=True, random_state=seed)

    results, cv_rmse = {}, {}
    for name, pipe in models.items():
        cv_rmse[name] = float(-cross_val_score(
            pipe, Xtr, ytr, cv=cv, scoring="neg_root_mean_squared_error").mean())
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        results[name] = {
            "mae": float(mean_absolute_error(yte, pred)),
            "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
            "r2": float(r2_score(yte, pred)),
            "cv_rmse_train": cv_rmse[name],
        }
    candidates = [n for n in models if n != "mean_baseline"]
    best = min(candidates, key=lambda n: cv_rmse[n])
    reason = (f"{best} has the lowest {CV_FOLDS}-fold cross-validated RMSE on the training "
              f"set ({cv_rmse[best]:.4f} tons); the test set was not used for selection.")
    report = {"baseline": {"mean_baseline": results["mean_baseline"]},
              "candidates": {n: results[n] for n in candidates},
              "selected_model": best, "selection_reason": reason}
    return report, models[best], models


def run_classification(split: dict, seed: int):
    """Compare majority baseline with Logistic Regression and RandomForest."""
    models = {
        "majority_baseline": _pipe(DummyClassifier(strategy="most_frequent")),
        "logistic_regression": _pipe(LogisticRegression(max_iter=1000, random_state=seed)),
        "random_forest": _pipe(RandomForestClassifier(n_estimators=300, random_state=seed)),
    }
    Xtr, ytr = split["X_train"], split["y_cls_train"]
    Xte, yte = split["X_test"], split["y_cls_test"]
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed)

    results, cv_f1 = {}, {}
    for name, pipe in models.items():
        cv_f1[name] = float(cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="f1").mean())
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        proba = pipe.predict_proba(Xte)[:, 1]
        auc = float(roc_auc_score(yte, proba)) if yte.nunique() == 2 else None
        results[name] = {
            "precision": float(precision_score(yte, pred, zero_division=0)),
            "recall": float(recall_score(yte, pred, zero_division=0)),
            "f1": float(f1_score(yte, pred, zero_division=0)),
            "roc_auc": auc,
            "confusion_matrix": confusion_matrix(yte, pred, labels=[0, 1]).tolist(),
            "cv_f1_train": cv_f1[name],
        }
    candidates = [n for n in models if n != "majority_baseline"]
    best = max(candidates, key=lambda n: cv_f1[n])
    reason = (f"{best} has the highest {CV_FOLDS}-fold cross-validated F1 on the training "
              f"set ({cv_f1[best]:.4f}); the test set was not used for selection.")
    report = {"confusion_matrix_labels": "rows = actual [0, 1]; columns = predicted [0, 1]; "
                                         "1 = high_risk",
              "baseline": {"majority_baseline": results["majority_baseline"]},
              "candidates": {n: results[n] for n in candidates},
              "selected_model": best, "selection_reason": reason}
    return report, models[best], models
