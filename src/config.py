"""Shared constants for the Musanze Harvest Intelligence Service."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NUMERIC_FEATURES = [
    "rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg", "field_size_ha",
]
BINARY_FEATURES = ["cooperative_support"]
CATEGORICAL_FEATURES = ["season"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

REG_TARGET = "yield_tons"
CLS_TARGET = "high_risk"
COLUMNS = FEATURES + [REG_TARGET, CLS_TARGET]

# Columns that must never be used as model inputs (target leakage).
FORBIDDEN_FEATURES = {"yield_tons", "yield_per_ha", "high_risk"}

# Inclusive valid ranges taken from the data-generation rules.
RANGES = {
    "rainfall_mm": (600.0, 1400.0),
    "avg_temp_c": (12.0, 26.0),
    "soil_moisture_pct": (20.0, 80.0),
    "fertilizer_kg": (0.0, 250.0),
    "field_size_ha": (0.2, 4.0),
    "cooperative_support": (0, 1),
    "yield_tons": (0.5, float("inf")),
}
SEASONS = ("A", "B")

TEST_SIZE = 0.20
CV_FOLDS = 5
K_RANGE = range(2, 7)

DISCLAIMER = (
    "Classroom decision-support prototype trained on synthetic data. It does not replace "
    "an agronomist and must not be used to decide access to finance, insurance or public "
    "services. A human must review every result."
)
