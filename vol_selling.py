"""Backtest: selling at-the-money BTC straddles on Deribit, always or only when implied vol is rich.

Two trades, both entered at 08:00 UTC and held to expiry:
  daily  - sell the straddle expiring the next day at 08:00, every day;
  weekly - sell the straddle expiring the following Friday, every Friday.
The entry price is Black-Scholes at the at-the-money implied vol of that expiry, fitted on the trades of
the hour before, minus a haircut for the bid-ask spread, minus Deribit fees. The strike is the spot at
entry. Settlement is the 07:30-08:00 average price, like Deribit's delivery price. P&L is in % of spot
for one BTC of notional. The weekly trade is also run delta-hedged every hour.

"Rich" means the entry implied vol is above the out-of-sample HAR forecast of realized vol from study.py.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

from surface import fit_expiry

ROOT = Path(__file__).parent
DATA, RES = ROOT / "data", ROOT / "results"
FEE_TRADE, FEE_SETTLE = 0.0003, 0.00015   # Deribit option fees per leg, fraction of the underlying
HEDGE_COST = 0.0003                        # per unit of BTC traded on the hedge
HAIRCUTS = [0.0, 0.01, 0.03]               # vol points given up to the spread when selling
THRESHOLDS = [None, 0.0, 0.05, 0.10]       # None = always sell; else sell if IV > HAR forecast + k


def settlement_prices(days):
    """Average of 1-minute Binance closes from 07:30 to 08:00 UTC, the window Deribit uses."""
    path = DATA / "settle.parquet"
    have = pd.read_parquet(path)["day"] if path.exists() else pd.Series(dtype="datetime64[ns, UTC]")
    rows = []
    for d in days:
        if (have == d).any():
            continue
        start = int((d + pd.Timedelta(hours=7, minutes=30)).timestamp() * 1000)
        for i in range(5):
            try:
                k = requests.get("https://api.binance.com/api/v3/klines", timeout=20,
                                 params=dict(symbol="BTCUSDT", interval="1m", startTime=start, limit=30)).json()
                break
            except requests.RequestException:
                time.sleep(2 * (i + 1))
        if k:
            rows.append(dict(day=d, settle=np.mean([float(x[4]) for x in k])))
    old = pd.read_parquet(path) if path.exists() else None
    out = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if rows else old
    out.to_parquet(path, index=False)
    return out.set_index("day")["settle"]


def entry_vols():
    """At-the-money IV at 08:00 for the next-day expiry and, on Fridays, the next-Friday expiry."""
    rows = []
    for path in sorted((DATA / "options").glob("*.parquet")):
        df = pd.read_parquet(path)
        snap = pd.to_datetime(df["snap"], unit="ms", utc=True)
        df = df[snap.dt.hour == 8]
        if df.empty:
            continue
        parts = df["instrument_name"].str.split("-", expand=True)
        expiry = pd.to_datetime(parts[1], format="%d%b%y", utc=True) + pd.Timedelta(hours=8)
        df = df.assign(expiry=expiry, K=parts[2].astype(float), sigma=df["iv"] / 100,
                       T=(expiry.astype("int64") // 10**6 - df["timestamp"]) / (365.25 * 86_400_000))
        df = df[df["sigma"].between(0.05, 3) & (df["T"] > 0)]
        d1 = (np.log(df["index_price"] / df["K"]) + 0.5 * df["sigma"] ** 2 * df["T"]) / (df["sigma"] * np.sqrt(df["T"]))
        df = df.assign(delta=norm.cdf(d1))
        df = df[df["delta"].between(0.08, 0.92)]
        for s, g in df.groupby("snap"):
            t = pd.Timestamp(s, unit="ms", tz="UTC")
            for kind, days in [("daily", 1), ("weekly", 7)]:
                if kind == "weekly" and t.dayofweek != 4:
                    continue
                e = g[g["expiry"] == t + pd.Timedelta(days=days)]
                if len(e) < 2:
                    continue
                fit = fit_expiry(e)
                if "atm" in fit and np.isfinite(fit["atm"]):
                    rows.append(dict(t=t, kind=kind, days=days, iv=fit["atm"], n=len(e)))
        print(path.stem, flush=True)
    return pd.DataFrame(rows)


def straddle(S, K, sigma, T):
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * norm.cdf(d2)
    return 2 * call - (S - K)  # call + put by parity, zero rates


def hedge_pnl(path, K, sigma, T0):
    """Short straddle hedged hourly: hold 2N(d1)-1 BTC, rebalanced on the hour, closed at settlement."""
    S = path.to_numpy()
    n = len(S) - 1
    taus = T0 - np.arange(n) / (365.25 * 24)
    taus = np.maximum(taus, 1e-6)
    d1 = (np.log(S[:-1] / K) + 0.5 * sigma**2 * taus) / (sigma * np.sqrt(taus))
    h = 2 * norm.cdf(d1) - 1
    gains = (h * np.diff(S)).sum()
    turnover = np.abs(np.diff(np.concatenate([[0], h, [0]]))).sum()
    return gains - HEDGE_COST * turnover * S.mean()


def run():
    RES.mkdir(exist_ok=True)
    vols_path = DATA / "entry_vols.parquet"
    if not vols_path.exists():
        entry_vols().to_parquet(vols_path, index=False)
    v = pd.read_parquet(vols_path)

    px = pd.read_parquet(DATA / "btc_1h.parquet")
    close = px.set_index(px["t"] + pd.Timedelta(hours=1))["close"]
    expiries = (v["t"] + pd.to_timedelta(v["days"], unit="D")).dt.normalize()
    settle = settlement_prices(sorted(expiries.unique()))

    har = {1: pd.read_parquet(RES / "vol_24h_HAR.parquet")["p"], 7: pd.read_parquet(RES / "vol_7d_HAR.parquet")["p"]}

    trades = []
    for r in v.itertuples():
        S0 = close.get(r.t)
        day_T = (r.t + pd.Timedelta(days=r.days)).normalize()
        ST = settle.get(day_T)
        if S0 is None or ST is None or np.isnan(S0) or np.isnan(ST):
            continue
        T = r.days / 365.25
        payoff = abs(ST - S0)
        fees = (2 * FEE_TRADE + FEE_SETTLE) * S0
        hp = har[r.days].get(r.t, np.nan)
        row = dict(t=r.t, kind=r.kind, iv=r.iv, har=np.exp(hp) if np.isfinite(hp) else np.nan,
                   move=(ST - S0) / S0, S0=S0)
        for hc in HAIRCUTS:
            prem = straddle(S0, S0, max(r.iv - hc, 0.05), T)
            row[f"pnl_{hc}"] = (prem - payoff - fees) / S0
            if r.kind == "weekly" and hc == 0.01:
                path = close.loc[r.t: day_T - pd.Timedelta(hours=1)]
                path = pd.concat([path, pd.Series([ST], index=[day_T])])
                row["pnl_hedged"] = (prem - payoff - fees + hedge_pnl(path, S0, r.iv - hc, T)) / S0
        trades.append(row)
    tr = pd.DataFrame(trades).set_index("t")
    tr.to_parquet(RES / "straddles.parquet")
    return tr


def stats(p, per_year):
    p = p.dropna()
    if len(p) < 5:
        return None
    eq = p.cumsum()
    rng = np.random.default_rng(0)
    x = p.to_numpy()
    block = 20 if per_year > 100 else 4
    boots = []
    for _ in range(2000):
        starts = rng.integers(0, max(1, len(x) - block), int(np.ceil(len(x) / block)))
        idx = (starts[:, None] + np.arange(block)).ravel()[:len(x)] % len(x)
        boots.append(x[idx].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n=len(p), mean=p.mean(), lo=lo, hi=hi, win=(p > 0).mean(), worst=p.min(),
                sharpe=p.mean() / p.std() * np.sqrt(per_year), total=p.sum(),
                max_dd=(eq.cummax() - eq).max(), per_year=p.sum() / len(p) * per_year)


def main():
    tr = run()
    out = []
    start = tr["har"].first_valid_index()
    for kind, per_year in [("daily", 365.25), ("weekly", 52.18)]:
        d = tr[tr["kind"] == kind]
        out.append(dict(kind=kind, rule="always, full period", pnl="pnl_0.01", **stats(d["pnl_0.01"], per_year)))
        d = d[(d.index >= start) & d["har"].notna()]  # same days for every rule
        cols = [f"pnl_{h}" for h in HAIRCUTS] + (["pnl_hedged"] if kind == "weekly" else [])
        for k in THRESHOLDS:
            sel = d if k is None else d[d["iv"] > d["har"] + k]
            for c in cols:
                s = stats(sel[c], per_year)
                if s:
                    # per_year counts only the days traded, so scale back to calendar time
                    s["per_year"] = sel[c].sum() / (len(d) / per_year)
                    out.append(dict(kind=kind, rule="always" if k is None else f"IV > HAR + {int(k * 100)}",
                                    pnl=c, **s))
    res = pd.DataFrame(out)
    print(res.round(4).to_string())
    print("iv vs har", tr.groupby("kind")[["iv", "har"]].mean().round(3).to_dict())
    with open(RES / "vol_selling.json", "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
