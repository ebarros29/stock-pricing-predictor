import time
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone


def _create_session():
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    session = requests.Session()
    session.verify = False
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    })
    return session


def _get_crumb(session) -> str:
    session.get("https://fc.yahoo.com/", timeout=15)
    time.sleep(1)
    for attempt in range(3):
        r = session.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=15)
        if r.status_code == 200:
            return r.text.strip()
        time.sleep(attempt + 1)
    raise RuntimeError(f"Failed to get crumb after 3 attempts (last: {r.status_code})")


def _date_to_unix(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def download_stock_data(symbol: str, start_date: str = None, end_date: str = None,
                        period: str = None) -> pd.DataFrame:
    session = _create_session()
    crumb = _get_crumb(session)

    if period:
        end = int(time.time())
        mapping = {"1y": 365, "6mo": 180, "3mo": 90, "1mo": 30, "5d": 5}
        days = mapping.get(period, 365)
        start = end - days * 86400
    else:
        start = _date_to_unix(start_date)
        end = _date_to_unix(end_date)

    url = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={start}&period2={end}&interval=1d&events=history&crumb={crumb}"
    )

    for attempt in range(5):
        r = session.get(url, timeout=15)
        if r.status_code == 200:
            break
        wait = (attempt + 1) * 3
        print(f"  Attempt {attempt + 1}: HTTP {r.status_code}, retrying in {wait}s...")
        time.sleep(wait)

    if r.status_code != 200:
        raise ValueError(f"Failed to fetch data for '{symbol}': HTTP {r.status_code}")

    data = r.json()
    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quotes = result["indicators"]["quote"][0]

    df = pd.DataFrame({
        "Date": pd.to_datetime(timestamps, unit="s"),
        "Open": quotes["open"],
        "High": quotes["high"],
        "Low": quotes["low"],
        "Close": quotes["close"],
        "Volume": quotes["volume"],
    })
    df = df.dropna().set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    if df.empty:
        raise ValueError(f"No data fetched for symbol '{symbol}'")
    return df


def save_raw_data(df: pd.DataFrame, symbol: str, output_dir: str = "outputs") -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = f"{output_dir}/{symbol}_raw.csv"
    df.to_csv(path)
    return path
