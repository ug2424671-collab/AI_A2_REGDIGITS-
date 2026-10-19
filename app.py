"""FastAPI service: uvicorn app:app --host 127.0.0.1 --port 8000"""
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.inference import ModelsNotFoundError, get_model_info, predict_validated
from src.schemas import FarmRecord, PredictionResponse

app = FastAPI(
    title="Musanze Harvest Intelligence Service",
    description="Classroom decision-support prototype (synthetic data). Human review required.",
    version="1.0.0",
)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    """Return a compact, readable 422 instead of the raw framework payload."""
    errors = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"),
               "message": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "invalid input", "details": errors})


@app.exception_handler(ModelsNotFoundError)
async def models_missing_handler(request: Request, exc: ModelsNotFoundError):
    return JSONResponse(status_code=503, content={"error": "models not available",
                                                  "details": str(exc)})


@app.get("/health")
def health():
    """Liveness plus confirmation that saved models can be loaded."""
    info = get_model_info()
    return {"status": "ok", "models_loaded": True,
            "verification_code": info["verification_code"]}


@app.get("/model-info")
def model_info():
    """Model, dataset and verification metadata."""
    return get_model_info()


@app.post("/predict", response_model=PredictionResponse)
def predict(record: FarmRecord):
    """Predict yield, risk, probability and cluster for one farm."""
    return predict_validated(record)
