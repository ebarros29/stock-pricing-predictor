import numpy as np
import pandas as pd
from src.data_collector import download_stock_data


def test_yahoo_chart_api_returns_expected_columns():
    df = download_stock_data("NVDA", period="1mo")
    assert not df.empty
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert isinstance(df.index, pd.DatetimeIndex)


def test_date_range_is_correct():
    df = download_stock_data("NVDA", start_date="2024-01-01", end_date="2024-03-31")
    assert df.index[0] >= pd.Timestamp("2024-01-01")
    assert df.index[-1] <= pd.Timestamp("2024-03-31")
    assert len(df) >= 30  # ~3 months of trading days


def test_no_nulls_in_data():
    df = download_stock_data("NVDA", period="6mo")
    assert df.isnull().sum().sum() == 0


def test_prices_are_positive():
    df = download_stock_data("NVDA", period="1y")
    for col in ["Open", "High", "Low", "Close"]:
        assert (df[col] > 0).all(), f"{col} has non-positive values"


def test_high_ge_low_and_open_close_in_range():
    df = download_stock_data("NVDA", period="3mo")
    assert (df["High"] >= df["Low"]).all()
    assert (df["High"] >= df["Open"]).all()
    assert (df["High"] >= df["Close"]).all()
    assert (df["Low"] <= df["Open"]).all()
    assert (df["Low"] <= df["Close"]).all()


def test_volume_positive():
    df = download_stock_data("NVDA", period="1mo")
    assert (df["Volume"] > 0).all()


def test_period_is_respected():
    df = download_stock_data("NVDA", period="1mo")
    n_days = (df.index[-1] - df.index[0]).days
    assert 15 <= n_days <= 45  # ~30 calendar days ± weekends
