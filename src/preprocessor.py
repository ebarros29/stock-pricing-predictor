import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib


class Preprocessor:
    def __init__(self, seq_length: int = 60):
        self.seq_length = seq_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_cols = None
        self.close_col_idx = None

    def fit(self, df: pd.DataFrame, feature_cols: list[str] = None):
        feature_cols = feature_cols or ["Close", "Open", "High", "Low", "Volume"]
        available = [c for c in feature_cols if c in df.columns]
        self.feature_cols = available
        self.close_col_idx = available.index("Close") if "Close" in available else 0
        self.scaler.fit(df[available].values)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(df[self.feature_cols].values)

    def inverse_transform_close(self, scaled_data: np.ndarray) -> np.ndarray:
        dummy = np.zeros((scaled_data.shape[0], len(self.feature_cols)))
        dummy[:, self.close_col_idx] = scaled_data.flatten()
        inverted = self.scaler.inverse_transform(dummy)
        return inverted[:, self.close_col_idx]

    def create_sequences(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(self.seq_length, len(data)):
            X.append(data[i - self.seq_length : i])
            y.append(data[i, self.close_col_idx])
        return np.array(X), np.array(y)

    def split_data(self, X: np.ndarray, y: np.ndarray, train_ratio=0.7, val_ratio=0.15) -> tuple:
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        return (
            X[:train_end], y[:train_end],
            X[train_end:val_end], y[train_end:val_end],
            X[val_end:], y[val_end:],
        )

    def save(self, path: str):
        joblib.dump({"scaler": self.scaler, "feature_cols": self.feature_cols,
                     "close_col_idx": self.close_col_idx, "seq_length": self.seq_length}, path)

    def predict_future(self, model, df: pd.DataFrame, days: int) -> np.ndarray:
        scaled = self.transform(df)
        window = scaled[-self.seq_length:].copy()
        n_features = len(self.feature_cols)
        current_seq = window.reshape(1, self.seq_length, n_features)
        predictions_scaled = []
        for _ in range(days):
            pred = model.predict(current_seq, verbose=0)[0, 0]
            predictions_scaled.append(float(pred))
            last_row = current_seq[0, -1, :].copy()
            last_row[self.close_col_idx] = pred
            new_seq = np.concatenate([current_seq[0, 1:, :], last_row.reshape(1, -1)], axis=0)
            current_seq = new_seq.reshape(1, self.seq_length, n_features)
        return self.inverse_transform_close(
            np.array(predictions_scaled).reshape(-1, 1))

    @staticmethod
    def load(path: str) -> "Preprocessor":
        data = joblib.load(path)
        pp = Preprocessor(seq_length=data["seq_length"])
        pp.scaler = data["scaler"]
        pp.feature_cols = data["feature_cols"]
        pp.close_col_idx = data["close_col_idx"]
        return pp
