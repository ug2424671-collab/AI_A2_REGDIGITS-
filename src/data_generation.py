"""Reproducible individual dataset generator (seed = last four registration digits)."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def generate_dataset(seed: int) -> pd.DataFrame:
    """Create the raw dataset including deliberate missing values and 3 duplicate rows."""
    rng = np.random.default_rng(seed)
    n = 240 + (seed % 41)

    rainfall = rng.uniform(600, 1400, n)
    temp = rng.uniform(12, 26, n)
    soil = rng.uniform(20, 80, n)
    fert = rng.uniform(0, 250, n)
    field = rng.uniform(0.2, 4.0, n)
    season = rng.choice(["A", "B"], size=n)
    support = rng.integers(0, 2, size=n)
    noise = rng.normal(0, 0.80, n)

    season_effect = np.where(season == "A", 0.50, -0.20)
    y = (0.004 * rainfall + 0.06 * soil + 0.02 * fert + 1.30 * field
         - 0.25 * np.abs(temp - 18) + 0.70 * support + season_effect + noise)
    y = np.clip(y, 0.50, None)

    yield_per_ha = y / field
    high_risk = (yield_per_ha < np.median(yield_per_ha)).astype(int)

    df = pd.DataFrame({
        "rainfall_mm": rainfall, "avg_temp_c": temp, "soil_moisture_pct": soil,
        "fertilizer_kg": fert, "field_size_ha": field, "season": season,
        "cooperative_support": support, "yield_tons": y, "high_risk": high_risk,
    })

    m = round(0.05 * n)
    all_positions = np.arange(n)
    soil_positions = rng.choice(all_positions, size=m, replace=False)
    remaining = np.setdiff1d(all_positions, soil_positions, assume_unique=True)
    fertilizer_positions = rng.choice(remaining, size=m, replace=False)
    df.loc[soil_positions, "soil_moisture_pct"] = np.nan
    df.loc[fertilizer_positions, "fertilizer_kg"] = np.nan

    eligible_dup_positions = np.setdiff1d(
        all_positions,
        np.union1d(soil_positions, fertilizer_positions),
        assume_unique=True,
    )
    dup_pos = rng.choice(eligible_dup_positions, size=3, replace=False)
    return pd.concat([df, df.iloc[dup_pos]], ignore_index=True)


def write_dataset(df: pd.DataFrame, path: Path) -> Path:
    """Write CSV with four decimals and fixed newlines (identical bytes on every OS)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.4f", lineterminator="\n")
    return path


def sha256_of_file(path: Path) -> str:
    """SHA-256 of the exact CSV bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_fingerprint(csv_path: Path, out_path: Path, verification_code: str) -> str:
    """Store the fingerprint text file and return the hash."""
    digest = sha256_of_file(csv_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"verification_code: {verification_code}\nfile: {Path(csv_path).name}\n"
        f"sha256: {digest}\n",
        encoding="utf-8",
    )
    return digest
