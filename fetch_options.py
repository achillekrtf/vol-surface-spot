"""Download Deribit BTC option trades in the hour before each 4-hour snapshot.

Each snapshot at time t (00:00, 04:00, ... UTC) keeps the trades printed in
[t - 1h, t), so every feature built from them is known at t. Output: one
parquet file per month in data/options/.
"""

import calendar
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

URL = "https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
OUT = Path(__file__).parent / "data" / "options"
STEP_H = 4
WINDOW_MS = 3_600_000
COLS = ["timestamp", "instrument_name", "iv", "index_price", "price", "mark_price", "amount", "direction"]

session = requests.Session()


def get(params, tries=8):
    for i in range(tries):
        try:
            r = session.get(URL, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()["result"]
        except requests.RequestException:
            pass
        time.sleep(min(30, 1.5 * 2**i))
    raise RuntimeError(f"failed {params}")


def window(t_ms):
    """All option trades in [t - 1h, t), paginating on has_more."""
    start, end, rows = t_ms - WINDOW_MS, t_ms - 1, []
    while True:
        r = get(dict(currency="BTC", kind="option", start_timestamp=start, end_timestamp=end,
                     count=1000, sorting="asc", include_old="true"))
        trades = r["trades"]
        rows += trades
        if not r["has_more"] or not trades:
            break
        start = trades[-1]["timestamp"] + 1
    df = pd.DataFrame(rows, columns=COLS) if rows else pd.DataFrame(columns=COLS)
    df["snap"] = t_ms
    return df.drop_duplicates()


def month(ym):
    path = OUT / f"{ym}.parquet"
    if path.exists():
        return
    first = pd.Timestamp(f"{ym}-01", tz="UTC")
    last = min(first + pd.offsets.MonthBegin(1), pd.Timestamp.now(tz="UTC").floor("4h"))
    snaps = pd.date_range(first, last, freq=f"{STEP_H}h", inclusive="left")
    with ThreadPoolExecutor(6) as ex:
        parts = list(ex.map(window, [int(s.timestamp() * 1000) for s in snaps]))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(path, index=False)
    print(ym, len(snaps), "snapshots", len(df), "trades", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    start, end = (sys.argv[1:3] + [None, None])[:2]
    months = pd.period_range(start or "2021-04", end or pd.Timestamp.now().strftime("%Y-%m"), freq="M")
    for m in months:
        month(str(m))
