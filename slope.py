"""Calendar, reverse calendar and straddle P&L by term-structure slope (1-day IV minus ~7-day IV).

Reprices the calendars from the stored implied vols (strike = spot, so everything scales with spot):
  calendar         sell the 1-day call, buy the ~7-day call, sell it back the next morning;
  reverse calendar buy the 1-day call, sell the ~7-day call, buy it back the next morning.
Slope buckets use thresholds fixed from the data before July 2022, so the test period never sets them.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from calendar_spread import YEAR, call, fee
from vol_selling import stats

ROOT = Path(__file__).parent
RES = ROOT / "results"
FIRST_TEST = pd.Timestamp("2022-07-01", tz="UTC")


def legs(r, h, direction):
    """direction +1 = calendar (short front), -1 = reverse calendar (long front). P&L in % of spot."""
    S1 = 1 + r.move
    Tf, Tb = 1 / YEAR, r.back_days / YEAR
    short_front = direction > 0
    front = call(1, 1, max(r.iv_front + (-h if short_front else h), 0.05), Tf)
    back_in = call(1, 1, max(r.iv_back + (h if short_front else -h), 0.05), Tb)
    back_out = call(S1, 1, max(r.iv_back_exit + (-h if short_front else h), 0.05), Tb - 1 / YEAR)
    expiry = max(S1 - 1, 0.0)
    gross = (front - expiry) - (back_in - back_out)
    fees = fee(1, front, 3e-4) + fee(1, back_in, 3e-4) + fee(S1, back_out, 3e-4) + (fee(S1, expiry, 1.5e-4) if expiry > 0 else 0)
    return direction * gross - fees


def main():
    c = pd.read_parquet(RES / "calendars.parquet").dropna(subset=["iv_front", "iv_back", "iv_back_exit"])
    c["slope"] = c["iv_front"] - c["iv_back"]
    for h in [0.0, 0.01]:
        c[f"cal_{h}"] = [legs(r, h, 1) for r in c.itertuples()]
        c[f"rev_{h}"] = [legs(r, h, -1) for r in c.itertuples()]
    edges = np.quantile(c.loc[c.index < FIRST_TEST, "slope"], [0.2, 0.4, 0.6, 0.8])
    test = c[c.index >= FIRST_TEST].copy()
    test["bucket"] = np.digitize(test["slope"], edges)
    out = []
    for b, g in test.groupby("bucket"):
        row = dict(bucket=int(b), n=len(g), slope=g["slope"].mean())
        for col in ["cal_0.0", "cal_0.01", "rev_0.0", "rev_0.01", "straddle_0.01"]:
            s = stats(g[col], 365.25)
            row[col] = s["mean"] if s else np.nan
            row[col + "_lo"], row[col + "_hi"] = (s["lo"], s["hi"]) if s else (np.nan, np.nan)
            row[col + "_total"] = g[col].sum()
        out.append(row)
    res = pd.DataFrame(out)
    print("bucket edges (vol pts):", (edges * 100).round(1))
    show = ["bucket", "n", "slope"] + [f"{k}{s}" for k in ["cal_0.01", "rev_0.01", "straddle_0.01"] for s in ["", "_lo", "_hi", "_total"]] + ["cal_0.0", "rev_0.0"]
    print(res[show].round(4).to_string())

    # the rule a trader would run: reverse calendar in the steepest bucket, calendar in the most inverted
    rule = pd.Series(0.0, index=test.index)
    rule[test["bucket"] == 4] = test["cal_0.01"]
    rule[test["bucket"] == 0] = test["rev_0.01"]
    traded = rule[test["bucket"].isin([0, 4])]
    print("slope rule:", {k: round(v, 4) if isinstance(v, float) else v for k, v in stats(traded, 365.25).items()})
    c.to_parquet(RES / "slope.parquet")
    with open(RES / "slope.json", "w") as f:
        json.dump(dict(edges=edges.tolist(), buckets=out, rule=stats(traded, 365.25)), f, indent=1, default=float)


if __name__ == "__main__":
    main()
