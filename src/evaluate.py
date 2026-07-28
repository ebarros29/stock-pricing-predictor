import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from src.data_collector import download_stock_data
from src.preprocessor import Preprocessor

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SYMBOL = "NVDA"
START_DATE = "2020-01-01"
END_DATE = "2026-07-24"
SEQ_LENGTH = 60
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, symbol: str, output_dir: str):
    plt.figure(figsize=(14, 6))
    plt.plot(y_true, label="Actual", linewidth=0.8)
    plt.plot(y_pred, label="Predicted", linewidth=0.8, alpha=0.85)
    plt.title(f"{symbol} — Actual vs Predicted Close Price")
    plt.xlabel("Test Samples")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    path = f"{output_dir}/{symbol}_predictions.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Plot saved: {path}")


def main():
    print(f"Loading data for {SYMBOL}...")
    df = download_stock_data(SYMBOL, START_DATE, END_DATE)

    print("Loading model and preprocessor...")
    model = load_model(f"{MODEL_DIR}/lstm_model.keras")
    pp = Preprocessor.load(f"{MODEL_DIR}/preprocessor.pkl")

    scaled = pp.transform(df)
    X_all, y_all = pp.create_sequences(scaled)
    _, _, _, _, X_test, y_test = pp.split_data(X_all, y_all)

    print("Predicting...")
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_test_real = pp.inverse_transform_close(y_test.reshape(-1, 1))
    y_pred_real = pp.inverse_transform_close(y_pred_scaled.reshape(-1, 1))

    metrics = compute_metrics(y_test_real, y_pred_real)
    print("\nTest Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    plot_predictions(y_test_real[-200:], y_pred_real[-200:], SYMBOL, OUTPUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
