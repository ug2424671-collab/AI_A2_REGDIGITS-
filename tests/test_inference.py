import pytest
from pydantic import ValidationError

from src.inference import predict_record


def test_valid_inference_returns_all_fields(valid_record):
    out = predict_record(valid_record)
    assert out["verification_code"] == "AI-A2-4671"
    assert out["yield_tons_estimate"] > 0
    assert out["high_risk_prediction"] in (0, 1)
    assert 0.0 <= out["high_risk_probability"] <= 1.0
    assert isinstance(out["cluster"], int)


def test_missing_field_rejected(valid_record):
    del valid_record["rainfall_mm"]
    with pytest.raises(ValidationError) as exc:
        predict_record(valid_record)
    assert "rainfall_mm" in str(exc.value)


def test_wrong_type_rejected(valid_record):
    valid_record["avg_temp_c"] = "hot"
    with pytest.raises(ValidationError):
        predict_record(valid_record)


@pytest.mark.parametrize("field,value", [
    ("rainfall_mm", 2500), ("soil_moisture_pct", 5), ("field_size_ha", 0.0),
    ("season", "C"), ("cooperative_support", 2),
])
def test_out_of_range_rejected(valid_record, field, value):
    valid_record[field] = value
    with pytest.raises(ValidationError):
        predict_record(valid_record)


def test_unknown_field_rejected(valid_record):
    valid_record["yield_tons"] = 10.0  # target must never be accepted as input
    with pytest.raises(ValidationError):
        predict_record(valid_record)
