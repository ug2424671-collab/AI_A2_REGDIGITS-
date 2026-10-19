from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info_has_verification_code():
    r = client.get("/model-info")
    assert r.status_code == 200
    assert r.json()["verification_code"] == "AI-A2-4671"


def test_predict_valid(valid_record):
    r = client.post("/predict", json=valid_record)
    assert r.status_code == 200
    body = r.json()
    assert body["verification_code"] == "AI-A2-4671"
    assert "yield_tons_estimate" in body and "cluster" in body


def test_predict_invalid_returns_useful_422(valid_record):
    valid_record["rainfall_mm"] = 5000
    valid_record["season"] = "Z"
    r = client.post("/predict", json=valid_record)
    assert r.status_code == 422
    fields = {d["field"] for d in r.json()["details"]}
    assert {"rainfall_mm", "season"} <= fields


def test_predict_missing_field_422(valid_record):
    del valid_record["field_size_ha"]
    r = client.post("/predict", json=valid_record)
    assert r.status_code == 422
