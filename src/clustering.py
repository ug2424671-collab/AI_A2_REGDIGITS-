"""KMeans farm segmentation (inputs only, no targets) with silhouette-based K choice."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .config import BINARY_FEATURES, K_RANGE, NUMERIC_FEATURES
from .preprocessing import build_preprocessor

_LABELS = {
    "rainfall_mm": "rainfall", "avg_temp_c": "temperature",
    "soil_moisture_pct": "soil moisture", "fertilizer_kg": "fertilizer use",
    "field_size_ha": "field size",
}


def _describe(cluster_df: pd.DataFrame, all_df: pd.DataFrame) -> str:
    """Plain-language profile from differences against the overall training average."""
    parts = []
    z = {c: (cluster_df[c].mean() - all_df[c].mean()) / all_df[c].std()
         for c in NUMERIC_FEATURES}
    for c in sorted(z, key=lambda k: -abs(z[k]))[:3]:
        if abs(z[c]) >= 0.5:
            parts.append(("higher " if z[c] > 0 else "lower ") + _LABELS[c])
    a_diff = (cluster_df["season"] == "A").mean() - (all_df["season"] == "A").mean()
    if abs(a_diff) >= 0.15:
        parts.append("mostly season A" if a_diff > 0 else "mostly season B")
    s_diff = cluster_df["cooperative_support"].mean() - all_df["cooperative_support"].mean()
    if abs(s_diff) >= 0.15:
        parts.append("mostly supported by the cooperative" if s_diff > 0
                     else "mostly without cooperative support")
    return ", ".join(parts) if parts else "close to the average farm on every input"


def run_clustering(split: dict, df: pd.DataFrame, seed: int):
    """Fit scaler + KMeans on training farms, score K=2..6, assign train and test farms."""
    Xtr, Xte = split["X_train"], split["X_test"]
    prep = build_preprocessor()
    Ztr = prep.fit_transform(Xtr)
    Zte = prep.transform(Xte)

    scores = {}
    for k in K_RANGE:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Ztr)
        scores[k] = float(silhouette_score(Ztr, labels))
    best_k = max(scores, key=scores.get)

    km = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit(Ztr)
    train_labels, test_labels = km.labels_, km.predict(Zte)

    profiles = {}
    for c in range(best_k):
        members = Xtr[train_labels == c]
        profiles[str(c)] = {
            "size_train": int(len(members)),
            "mean_numeric_inputs": {f: round(float(members[f].mean()), 3)
                                    for f in NUMERIC_FEATURES},
            "share_season_A": round(float((members["season"] == "A").mean()), 3),
            "share_cooperative_support": round(float(members[BINARY_FEATURES[0]].mean()), 3),
            "description": _describe(members, Xtr),
            "posthoc_mean_yield_tons": round(float(
                df.loc[members.index, "yield_tons"].mean()), 3),
            "posthoc_high_risk_rate": round(float(
                df.loc[members.index, "high_risk"].mean()), 3),
        }
    report = {
        "silhouette_by_k": {str(k): v for k, v in scores.items()},
        "selected_k": best_k,
        "selection_reason": (f"K={best_k} has the highest silhouette score "
                             f"({scores[best_k]:.4f}) among K=2..6."),
        "profiles": profiles,
        "note_on_targets": ("Clusters were fitted on input features only. The 'posthoc_*' "
                            "fields describe each cluster afterwards and were not used to fit."),
        "interpretation": (
            "Silhouette values are modest because the synthetic inputs are independent "
            "uniform draws, so farms form no strongly separated natural groups. The "
            "clusters are therefore soft groupings of similar growing conditions that a "
            "field officer can use to plan visits, not hard categories."),
    }
    return report, prep, km, Ztr, train_labels, test_labels
