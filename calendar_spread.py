"""Backtest: one-day ATM call calendar on Deribit BTC options.

Every day at 08:00 UTC: sell the call expiring the next morning and buy the call at the same strike on
the expiry closest to 7 days out (5 to 10 days). The next morning the short call settles and the long
call is sold back. Strike = spot at entry. Prices are Black-Scholes at implied vols read from smiles
fitted on the trades of the hour before entry and before exit; each leg gives up `h` vol points to the
spread every time it trades. Deribit fees: 0.03% of the underlying per trade and 0.015% at settlement,
each capped at 12.5% of the option price. P&L in % of spot for one BTC per leg.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from vol_selling import settlement_prices, stats

ROOT = Path(__file__).parent
DATA, RES = ROOT / "data", ROOT / "results"
YEAR = 365.25
HAIRCUTS = [0.0, 0.01, 0.03]


def smiles():
    """Smile per (08:00 snapshot, expiry): quadratic in call delta, or a flat ATM level if wings are thin."""
    rows = []
    for path in sorted((DATA / "options").glob("*.parquet")):
        df = pd.read_parquet(path)
        t = pd.to_datetime(df["snap"], unit="ms", utc=True)
        df = df[t.dt.hour == 8]
        parts = df["instrument_name"].str.split("-", expand=True)
        expiry = pd.to_datetime(parts[1], format="%d%b%y", utc=True) + pd.Timedelta(hours=8)
        df = df.assign(expiry=expiry, K=parts[2].astype(float), sigma=df["iv"] / 100,
                       T=(expiry.astype("int64") // 10**6 - df["timestamp"]) / (YEAR * 86_400_000))
        df = df[df["sigma"].between(0.05, 3) & (df["T"] > 0.5 / YEAR) & (df["T"] < 12 / YEAR)]
        d1 = (np.log(df["index_price"] / df["K"]) + 0.5 * df["sigma"] ** 2 * df["T"]) / (df["sigma"] * np.sqrt(df["T"]))
        df = df.assign(x=norm.cdf(d1) - 0.5)
        df = df[df["x"].abs() <= 0.42]
        for (s, e), g in df.groupby(["snap", "expiry"]):
            x, y, w = g["x"].to_numpy(), g["sigma"].to_numpy(), np.sqrt(g["amount"].to_numpy())
            near = np.abs(x) < 0.15
            row = dict(t=pd.Timestamp(s, unit="ms", tz="UTC"), expiry=e, a=np.nan, b=0.0, c=0.0)
            if len(g) >= 6 and (x <= -0.18).sum() >= 2 and (x >= 0.18).sum() >= 2 and near.any():
                a, b, c = np.polynomial.polynomial.polyfit(x, y, 2, w=w)
                if 0.1 < a < 3 and abs(c) < 3:
                    row.update(a=a, b=b, c=c)
            if np.isnan(row["a"]) and near.sum() >= 2:
                row["a"] = np.average(y[near], weights=w[near])
            if np.isfinite(row["a"]):
                rows.append(row)
        print(path.stem, flush=True)
    return pd.DataFrame(rows)


def iv_at(sm, S, K, T):
    """Implied vol at strike K: solve delta(K) on the smile by fixed point."""
    sig = sm["a"]
    for _ in range(4):
        x = norm.cdf((np.log(S / K) + 0.5 * sig**2 * T) / (sig * np.sqrt(T))) - 0.5
        x = np.clip(x, -0.42, 0.42)
        sig = max(0.05, sm["a"] + sm["b"] * x + sm["c"] * x * x)
    return sig


def call(S, K, sig, T):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + 0.5 * sig**2 * T) / (sig * np.sqrt(T))
    return S * norm.cdf(d1) - K * norm.cdf(d1 - sig * np.sqrt(T))


def fee(S, price, rate):
    return min(rate * S, 0.125 * price)


def main():
    path = DATA / "smiles_08.parquet"
    if not path.exists():
        smiles().to_parquet(path, index=False)
    sm = pd.read_parquet(path)
    sm = sm.set_index(["t", "expiry"]).sort_index()

    px = pd.read_parquet(DATA / "btc_1h.parquet")
    close = px.set_index(px["t"] + pd.Timedelta(hours=1))["close"]
    days = sorted(sm.index.get_level_values("t").unique())
    settle = settlement_prices([(d + pd.Timedelta(days=1)).normalize() for d in days])
    har = pd.read_parquet(RES / "vol_24h_HAR.parquet")["p"]
    straddles = pd.read_parquet(RES / "straddles.parquet")
    straddles = straddles[straddles["kind"] == "daily"]

    rows = []
    for t in days:
        t1 = t + pd.Timedelta(days=1)
        if t not in sm.index.get_level_values(0) or t1 not in sm.index.get_level_values(0):
            continue
        today, tomorrow = sm.loc[t], sm.loc[t1]
        front_e = t + pd.Timedelta(days=1)
        backs = [e for e in today.index if 5 <= (e - t).days <= 10]
        if front_e not in today.index or not backs:
            continue
        back_e = min(backs, key=lambda e: abs((e - t).days - 7))
        if back_e not in tomorrow.index:
            continue
        S0, S1 = close.get(t), settle.get(t1.normalize())
        if S0 is None or S1 is None or np.isnan(S0) or np.isnan(S1):
            continue
        K = S0
        Tf, Tb = 1 / YEAR, (back_e - t).total_seconds() / 86400 / YEAR
        iv_f = iv_at(today.loc[front_e], S0, K, Tf)
        iv_b = iv_at(today.loc[back_e], S0, K, Tb)
        iv_b1 = iv_at(tomorrow.loc[back_e], S1, K, Tb - 1 / YEAR)
        row = dict(t=t, iv_front=iv_f, iv_back=iv_b, iv_back_exit=iv_b1, move=S1 / S0 - 1,
                   har=np.exp(har[t]) if t in har.index else np.nan, back_days=(back_e - t).days)
        for h in HAIRCUTS:
            sold = call(S0, K, max(iv_f - h, 0.05), Tf)
            bought = call(S0, K, iv_b + h, Tb)
            resold = call(S1, K, max(iv_b1 - h, 0.05), Tb - 1 / YEAR)
            paid_at_expiry = max(S1 - K, 0.0)
            fees = (fee(S0, sold, 3e-4) + fee(S0, bought, 3e-4) + fee(S1, resold, 3e-4)
                    + (fee(S1, paid_at_expiry, 1.5e-4) if paid_at_expiry > 0 else 0.0))
            row[f"pnl_{h}"] = (sold - paid_at_expiry - bought + resold - fees) / S0
            row[f"debit_{h}"] = (bought - sold) / S0
        st = straddles["pnl_0.01"].get(t, np.nan)
        row["straddle_0.01"] = st
        row["straddle_0.03"] = straddles["pnl_0.03"].get(t, np.nan)
        rows.append(row)
    cal = pd.DataFrame(rows).set_index("t")
    cal.to_parquet(RES / "calendars.parquet")

    test = cal[(cal.index >= pd.Timestamp("2022-07-01", tz="UTC")) & cal["har"].notna()]
    rules = {"always": pd.Series(True, index=test.index),
             "IV1d > HAR + 5": test["iv_front"] > test["har"] + 0.05,
             "IV1d > IV7d": test["iv_front"] > test["iv_back"],
             "both": (test["iv_front"] > test["har"] + 0.05) & (test["iv_front"] > test["iv_back"])}
    out = []
    for name, sel in rules.items():
        for c in ["pnl_0.0", "pnl_0.01", "pnl_0.03", "straddle_0.01", "straddle_0.03"]:
            s = stats(test.loc[sel, c], 365.25)
            if s:
                out.append(dict(rule=name, pnl=c, **s))
    res = pd.DataFrame(out)
    print(res[["rule", "pnl", "n", "mean", "lo", "hi", "win", "worst", "total", "max_dd"]].round(4).to_string())
    print("mean debit (1pt)", test["debit_0.01"].mean().round(4), "iv front/back", test[["iv_front", "iv_back"]].mean().round(3).to_dict(),
          "share front>back", (test.iv_front > test.iv_back).mean().round(3), "days", len(test))
    with open(RES / "calendars.json", "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
