"""Train once into a temporary project root so tests never touch submitted artifacts."""
import pytest

from src.pipeline import run_pipeline

TEST_ID = "ug2424671@ines.ac.rw"
TEST_SEED = 4671


@pytest.fixture(scope="session")
def trained_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("project")
    run_pipeline(TEST_ID, TEST_SEED, root)
    return root


@pytest.fixture(autouse=True)
def use_trained_models(trained_root, monkeypatch):
    monkeypatch.setenv("AI_A2_ROOT", str(trained_root))


@pytest.fixture()
def valid_record():
    return {"rainfall_mm": 950.0, "avg_temp_c": 18.5, "soil_moisture_pct": 55.0,
            "fertilizer_kg": 120.0, "field_size_ha": 1.8, "season": "A",
            "cooperative_support": 1}
