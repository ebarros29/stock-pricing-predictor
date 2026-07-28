import os
import time
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import prometheus_client

from src.data_collector import download_stock_data
from src.preprocessor import Preprocessor

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODEL_DIR = "models"

# ── Custom Prometheus metrics ───────────────────────────────────

PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total number of model predictions made",
    ["endpoint", "symbol"],
)

PREDICTION_DURATION = Histogram(
    "model_prediction_duration_seconds",
    "Model inference duration in seconds",
    ["endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

MODEL_LOAD_TIME = Gauge(
    "model_load_duration_seconds",
    "Time taken to load the TensorFlow model from disk",
)

# ── Load model & preprocessor ───────────────────────────────────

_t0 = time.perf_counter()
from tensorflow.keras.models import load_model
model = load_model(f"{MODEL_DIR}/lstm_model.keras")
preprocessor = Preprocessor.load(f"{MODEL_DIR}/preprocessor.pkl")
MODEL_LOAD_TIME.set(time.perf_counter() - _t0)

app = FastAPI(title="Stock Price Predictor", version="2.0.0")

Instrumentator(registry=prometheus_client.REGISTRY).instrument(app).expose(app)

# ── Schemas ──────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    symbol: str = Field(default="NVDA", description="Stock ticker symbol")
    days: int = Field(default=7, ge=1, le=90, description="Number of future days to predict")


class PredictResponse(BaseModel):
    symbol: str
    predicted_close: list[float]
    currency: str = "USD"


class OHLCVRecord(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class PredictFromDataRequest(BaseModel):
    historical_data: list[OHLCVRecord] = Field(
        ..., min_length=60,
        description="Historical OHLCV records. Must contain at least 60 rows."
    )
    days: int = Field(default=7, ge=1, le=90, description="Number of future days to predict")


class PredictFromDataResponse(BaseModel):
    predicted_close: list[float]
    currency: str = "USD"


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "model": "LSTM", "symbol": "NVDA"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    df = download_stock_data(req.symbol, period="1y")
    if len(df) < preprocessor.seq_length:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {preprocessor.seq_length} days of data, got {len(df)}"
        )

    _t0 = time.perf_counter()
    prices = preprocessor.predict_future(model, df, req.days)
    elapsed = time.perf_counter() - _t0

    PREDICTION_COUNT.labels(endpoint="/predict", symbol=req.symbol).inc(req.days)
    PREDICTION_DURATION.labels(endpoint="/predict").observe(elapsed)

    return PredictResponse(
        symbol=req.symbol,
        predicted_close=[round(float(p), 2) for p in prices]
    )


@app.post("/predict-from-data", response_model=PredictFromDataResponse)
def predict_from_data(req: PredictFromDataRequest):
    rows = [r.model_dump() for r in req.historical_data]
    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "date": "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    missing = set(preprocessor.feature_cols) - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing columns: {sorted(missing)}. Required: {preprocessor.feature_cols}"
        )

    if len(df) < preprocessor.seq_length:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {preprocessor.seq_length} rows, got {len(df)}"
        )

    _t0 = time.perf_counter()
    prices = preprocessor.predict_future(model, df, req.days)
    elapsed = time.perf_counter() - _t0

    PREDICTION_COUNT.labels(endpoint="/predict-from-data", symbol="custom").inc(req.days)
    PREDICTION_DURATION.labels(endpoint="/predict-from-data").observe(elapsed)

    return PredictFromDataResponse(
        predicted_close=[round(float(p), 2) for p in prices]
    )
