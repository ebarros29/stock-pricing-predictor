import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

from src.data_collector import download_stock_data
from src.preprocessor import Preprocessor

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODEL_DIR = "models"

from tensorflow.keras.models import load_model
model = load_model(f"{MODEL_DIR}/lstm_model.keras")
preprocessor = Preprocessor.load(f"{MODEL_DIR}/preprocessor.pkl")

app = FastAPI(title="Stock Price Predictor", version="2.0.0")

Instrumentator().instrument(app).expose(app)

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
    prices = preprocessor.predict_future(model, df, req.days)
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

    prices = preprocessor.predict_future(model, df, req.days)
    return PredictFromDataResponse(
        predicted_close=[round(float(p), 2) for p in prices]
    )
