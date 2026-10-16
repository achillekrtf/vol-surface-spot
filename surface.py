"""Turn the option trades around each snapshot into a small set of surface features.

For every expiry with enough trades, implied vol is fitted as a quadratic in
call delta (x = delta - 0.5), weighted by trade size. That gives at-the-money
vol, the 25-delta risk reversal (call wing minus put wing) and the 25-delta
butterfly. Expiries are then interpolated to fixed tenors of 7, 30 and 90 days.
Output: data/surface.parquet, one row per snapshot.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

DATA = Path(__file__).parent / "data"
YEAR_MS = 365.25 * 86_400_000
TENORS = {"7": 7 / 365.25, "30": 30 / 365.25, "90": 90 / 365.25}
MIN_T = 2 / 365.25


def parse(df):
    parts = df["instrument_name"].str.split("-", expand=True)
    expiry = pd.to_datetime(parts[1], format="%d%b%y", utc=True) + pd.Timedelta(hours=8)
    df = df.assign(
        expiry=expiry,
        K=parts[2].astype(float),
        is_call=parts[3].eq("C"),
        T=(expiry.astype("int64") // 10**6 - df["timestamp"]) / YEAR_MS,
        sigma=df["iv"] / 100,
    )
    df = df[(df["T"] > MIN_T) & df["sigma"].between(0.05, 3)]
    d1 = (np.log(df["index_price"] / df["K"]) + 0.5 * df["sigma"] ** 2 * df["T"]) / (df["sigma"] * np.sqrt(df["T"]))
    df = df.assign(delta=norm.cdf(d1))
    return df[df["delta"].between(0.08, 0.92)]


def fit_expiry(g):
    """Quadratic smile in delta; wings only when both sides traded."""
    x = g["delta"].to_numpy() - 0.5
    y = g["sigma"].to_numpy()
    w = np.sqrt(g["amount"].to_numpy())
    near = np.abs(x) < 0.15
    # the 25-delta points must sit inside the traded range, not be extrapolated
    calls, puts = (x <= -0.18).sum(), (x >= 0.18).sum()
    out = {"T": g["T"].mean(), "n": len(g)}
    if len(g) >= 6 and calls >= 2 and puts >= 2 and near.sum() >= 1:
        a, b, c = np.polynomial.polynomial.polyfit(x, y, 2, w=w)
        f = lambda z: a + b * z + c * z * z
        rr, bf = f(-0.25) - f(0.25), (f(-0.25) + f(0.25)) / 2 - a
        if abs(rr) < 0.4 and -0.03 < bf < 0.3 and 0.1 < a < 3:
            out.update(atm=a, rr=rr, bf=bf)
            return out
    if near.sum() >= 2:
        out.update(atm=np.average(y[near], weights=w[near]))
    return out


def interp(fits, col, T, total_var=False):
    if col not in fits:
        return np.nan
    f = fits.dropna(subset=[col]).sort_values("T")
    if len(f) == 0 or T < f["T"].iloc[0] * 0.6 or T > f["T"].iloc[-1] * 1.4:
        return np.nan
    v = f[col].to_numpy()
    if total_var:
        v = v**2 * f["T"].to_numpy()
    y = np.interp(T, f["T"].to_numpy(), v)
    return np.sqrt(y / T) if total_var else y


def snapshot(g):
    fits = pd.DataFrame([fit_expiry(e) for _, e in g.groupby("expiry") if len(e) >= 2])
    row = {"n_trades": len(g), "spot_index": g["index_price"].iloc[-1]}
    if len(fits):
        for name, T in TENORS.items():
            row[f"atm{name}"] = interp(fits, "atm", T, total_var=True)
            row[f"rr{name}"] = interp(fits, "rr", T)
            row[f"bf{name}"] = interp(fits, "bf", T)
    # taker flow in the hour: bought calls minus bought puts, in BTC notional
    sign = np.where(g["direction"] == "buy", 1, -1)
    row["flow_call"] = (sign * g["amount"] * g["is_call"]).sum()
    row["flow_put"] = (sign * g["amount"] * ~g["is_call"]).sum()
    row["vega_flow"] = (sign * g["amount"] * np.sqrt(g["T"])).sum()
    return row


def main():
    rows = []
    for path in sorted((DATA / "options").glob("*.parquet")):
        df = parse(pd.read_parquet(path))
        for snap, g in df.groupby("snap"):
            rows.append({"t": snap, **snapshot(g)})
        print(path.stem, flush=True)
    out = pd.DataFrame(rows)
    out["t"] = pd.to_datetime(out["t"], unit="ms", utc=True)
    out.sort_values("t").to_parquet(DATA / "surface.parquet", index=False)
    print(out.describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
