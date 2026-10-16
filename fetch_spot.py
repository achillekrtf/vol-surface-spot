"""Download hourly BTCUSDT candles from Binance and hourly DVOL from Deribit."""

import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).parent / "data"
START = int(pd.Timestamp("2021-01-01", tz="UTC").timestamp() * 1000)


def binance():
    rows, t = [], START
    now = int(time.time() * 1000)
    while t < now:
        r = requests.get("https://api.binance.com/api/v3/klines",
                         params=dict(symbol="BTCUSDT", interval="1h", startTime=t, limit=1000), timeout=30).json()
        if not r:
            break
        rows += r
        t = r[-1][0] + 3_600_000
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["t", "open", "high", "low", "close", "volume"]
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.astype({c: float for c in ["open", "high", "low", "close", "volume"]})
    df.drop_duplicates("t").to_parquet(OUT / "btc_1h.parquet", index=False)
    print("binance", len(df), df.t.min(), df.t.max())


def dvol():
    rows, end = [], int(time.time() * 1000)
    while True:
        r = requests.get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                         params=dict(currency="BTC", start_timestamp=START, end_timestamp=end, resolution=3600),
                         timeout=30).json()["result"]
        rows += r["data"]
        if not r.get("continuation") or not r["data"]:
            break
        end = r["continuation"]
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close"])
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df.drop_duplicates("t").sort_values("t").to_parquet(OUT / "dvol_1h.parquet", index=False)
    print("dvol", len(df), df.t.min(), df.t.max())


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    binance()
    dvol()
