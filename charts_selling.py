"""Charts for the straddle-selling article."""

import json
import sys
from pathlib import Path

import pandas as pd

from charts import AMBER, BG, BLUE, FG, MUTED, SLATE, plt, save  # noqa: F401  (also applies the theme)
import charts

ROOT = Path(__file__).parent
RES = ROOT / "results"
charts.OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "charts"
charts.OUT.mkdir(parents=True, exist_ok=True)

tr = pd.read_parquet(RES / "straddles.parquet")
d = tr[(tr["kind"] == "daily") & tr["har"].notna()]

# 1. cumulative P&L of the daily straddle by rule
rules = [("Always sell", None, SLATE), ("IV above the HAR forecast", 0.0, "#7fb6ff"),
         ("IV 5 points above", 0.05, BLUE), ("IV 10 points above", 0.10, AMBER)]
fig, ax = plt.subplots(figsize=(10, 4.6))
for lab, k, col in rules:
    p = d["pnl_0.01"] if k is None else d["pnl_0.01"].where(d["iv"] > d["har"] + k, 0.0)
    ax.plot(p.index, p.cumsum() * 100, color=col, lw=1.8 if k else 1.4, label=lab)
ax.axhline(0, color=FG, lw=0.8, alpha=0.6)
ax.set_ylabel("Cumulative P&L, % of 1 BTC notional", color=MUTED)
ax.legend(loc="upper left")
ax.set_title("Selling a one-day BTC straddle every morning, or only when it looks expensive", loc="left",
             color=FG, fontsize=13, pad=14)
save(fig, "straddle-cumulative")

# 2. sensitivity to the spread given up when selling
res = pd.DataFrame(json.load(open(RES / "vol_selling.json")))
res = res[(res["kind"] == "daily") & (res["rule"] != "always, full period")]
order = ["always", "IV > HAR + 0", "IV > HAR + 5", "IV > HAR + 10"]
names = ["Always", "IV > HAR", "IV > HAR + 5", "IV > HAR + 10"]
cuts = [("pnl_0.0", "Sold at the traded IV", "#7fb6ff"), ("pnl_0.01", "1 vol point given up", BLUE),
        ("pnl_0.03", "3 vol points given up", AMBER)]
fig, ax = plt.subplots(figsize=(10, 4.4))
w = 0.26
for j, (col_name, lab, col) in enumerate(cuts):
    for i, rule in enumerate(order):
        row = res[(res["rule"] == rule) & (res["pnl"] == col_name)].iloc[0]
        x = i + (j - 1) * w
        ax.bar(x, row["total"] * 100, w * 0.9, color=col, label=lab if i == 0 else None)
ax.axhline(0, color=FG, lw=0.8, alpha=0.6)
ax.set_xticks(range(4), names)
ax.set_ylabel("Total P&L, Jul 2022 to Oct 2026, % of notional", color=MUTED)
ax.legend(loc="upper left")
ax.set_title("How much the spread matters", loc="left", color=FG, fontsize=13, pad=14)
save(fig, "straddle-spread")

# 3. weekly straddle, naked vs delta-hedged
wk = tr[(tr["kind"] == "weekly") & tr["har"].notna()]
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.plot(wk.index, wk["pnl_0.01"].cumsum() * 100, color=SLATE, lw=1.6, label="Weekly straddle, not hedged")
ax.plot(wk.index, wk["pnl_hedged"].cumsum() * 100, color=BLUE, lw=1.8, label="Weekly straddle, delta-hedged hourly")
ax.axhline(0, color=FG, lw=0.8, alpha=0.6)
ax.set_ylabel("Cumulative P&L, % of notional", color=MUTED)
ax.legend(loc="upper left")
ax.set_title("Selling the weekly straddle every Friday", loc="left", color=FG, fontsize=13, pad=14)
save(fig, "straddle-weekly")
print("done")

# 4. ML filter vs the fixed rule
dec = pd.read_parquet(RES / "ml_selling_decisions.parquet")
t = d.loc[dec.index]
lines = [("Always sell", pd.Series(True, index=t.index), SLATE),
         ("Rule: IV 5 points above HAR", t["iv"] > t["har"] + 0.05, BLUE),
         ("ML, linear (ridge)", dec["everything | ridge"], AMBER),
         ("ML, gradient-boosted trees", dec["everything | gbm"], "#7fb6ff")]
fig, ax = plt.subplots(figsize=(10, 4.6))
for lab, sell, col in lines:
    p = t["pnl_0.01"].where(sell, 0.0)
    ax.plot(p.index, p.cumsum() * 100, color=col, lw=1.8, label=lab)
ax.axhline(0, color=FG, lw=0.8, alpha=0.6)
ax.set_ylabel("Cumulative P&L, % of 1 BTC notional", color=MUTED)
ax.legend(loc="upper left")
ax.set_title("Letting a model decide when to sell the one-day straddle", loc="left", color=FG, fontsize=13, pad=14)
save(fig, "straddle-ml")
print("ml chart")
