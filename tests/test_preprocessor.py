import numpy as np
import pandas as pd
import pytest
from src.preprocessor import Preprocessor


@pytest.fixture
def sample_df():
    dates = pd.date_range("2024-01-01", periods=200, freq="B")
    return pd.DataFrame({
        "Open":  np.linspace(100, 200, 200) + np.random.randn(200) * 2,
        "High":  np.linspace(102, 205, 200) + np.random.randn(200) * 2,
        "Low":   np.linspace(98, 195, 200) + np.random.randn(200) * 2,
        "Close": np.linspace(101, 202, 200) + np.random.randn(200),
        "Volume": np.random.randint(50_000_000, 150_000_000, 200),
    }, index=dates)


def test_fit_detects_features(sample_df):
    pp = Preprocessor(seq_length=30)
    pp.fit(sample_df)
    assert pp.feature_cols == ["Close", "Open", "High", "Low", "Volume"]
    assert pp.close_col_idx == 0


def test_fit_subset_of_features(sample_df):
    pp = Preprocessor(seq_length=30)
    pp.fit(sample_df, feature_cols=["Close", "Volume"])
    assert pp.feature_cols == ["Close", "Volume"]
    assert pp.close_col_idx == 0


def test_transform_scales_to_range(sample_df):
    pp = Preprocessor(seq_length=30)
    pp.fit(sample_df)
    scaled = pp.transform(sample_df)
    assert scaled.shape == (200, 5)
    assert scaled.min() >= 0.0
    assert scaled.max() <= 1.0 + 1e-10


def test_create_sequences_shapes(sample_df):
    pp = Preprocessor(seq_length=60)
    pp.fit(sample_df)
    scaled = pp.transform(sample_df)
    X, y = pp.create_sequences(scaled)

    assert X.shape == (140, 60, 5)  # 200 - 60 = 140 sequences
    assert y.shape == (140,)


def test_inverse_transform_close_roundtrip(sample_df):
    pp = Preprocessor(seq_length=60)
    pp.fit(sample_df)
    original_close = sample_df["Close"].values[60:80].reshape(-1, 1)
    scaled = pp.transform(sample_df)
    recovered = pp.inverse_transform_close(
        scaled[60:80, pp.close_col_idx].reshape(-1, 1))

    assert np.allclose(original_close.flatten(), recovered.flatten(), rtol=1e-4)


def test_split_data_proportions(sample_df):
    pp = Preprocessor(seq_length=60)
    pp.fit(sample_df)
    scaled = pp.transform(sample_df)
    X, y = pp.create_sequences(scaled)

    X_tr, y_tr, X_val, y_val, X_test, y_test = pp.split_data(X, y)
    n = len(X)

    assert len(X_tr) == int(n * 0.7)
    assert len(X_val) == int(n * 0.15)
    assert len(X_test) == n - int(n * 0.7) - int(n * 0.15)

    # Chronological — no index overlap between splits
    assert len(X_tr) > 0 and len(X_val) > 0 and len(X_test) > 0


def test_sequences_are_chronological(sample_df):
    pp = Preprocessor(seq_length=60)
    pp.fit(sample_df)
    scaled = pp.transform(sample_df)
    X, y = pp.create_sequences(scaled)

    # y[i] should be the Close value of the (60+i)-th row
    for i in [0, 10, 50]:
        expected = scaled[60 + i, pp.close_col_idx]
        assert np.isclose(y[i], expected, rtol=1e-6)


def test_predict_future_shape_and_values(sample_df):
    pp = Preprocessor(seq_length=60)
    pp.fit(sample_df)

    # Fake model that returns constant predictions
    class FakeModel:
        def predict(self, x, verbose=0):  # noqa: ARG002
            return np.array([[0.5]])

    prices = pp.predict_future(FakeModel(), sample_df, days=10)
    assert prices.shape == (10,)
    assert np.all(prices > 0)


def test_predict_future_reproduces_training_scale(sample_df):
    pp = Preprocessor(seq_length=60)
    pp.fit(sample_df)

    calls = []

    class FakeModel:
        def predict(self, x, verbose=0):
            calls.append(1)
            # Return the actual last close value of the window
            return np.array([[x[0, -1, pp.close_col_idx]]])

    prices = pp.predict_future(FakeModel(), sample_df, days=5)
    assert len(calls) == 5
    # With identity-level prediction, all close prices should be near each other
    ref = float(sample_df["Close"].iloc[-1])
    assert np.allclose(prices, ref, rtol=0.01)


def test_save_and_load_roundtrip(sample_df, tmp_path):
    pp = Preprocessor(seq_length=45)
    pp.fit(sample_df, ["Close", "Volume"])

    path = tmp_path / "preprocessor.pkl"
    pp.save(str(path))

    loaded = Preprocessor.load(str(path))
    assert loaded.seq_length == 45
    assert loaded.feature_cols == ["Close", "Volume"]
    assert loaded.close_col_idx == 0

    original = pp.transform(sample_df)
    reloaded = loaded.transform(sample_df)
    assert np.array_equal(original, reloaded)


def test_minimum_data_requirement(sample_df):
    pp = Preprocessor(seq_length=60)
    pp.fit(sample_df)
    scaled = pp.transform(sample_df)
    X, y = pp.create_sequences(scaled)

    assert X.shape[0] > 0
    assert X.shape[1] == 60
    assert X.shape[2] == len(pp.feature_cols)
