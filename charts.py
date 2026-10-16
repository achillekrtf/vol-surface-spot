"""Charts for the article, in the site's dark theme."""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
RES = ROOT / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "charts"
OUT.mkdir(parents=True, exist_ok=True)

BG, FG, MUTED, GRID, AXIS = "#10161e", "#e6ebf2", "#8a96a8", "#1c2531", "#2c3746"
BLUE, AMBER, SLATE = "#4f9bff", "#f2a65a", "#586374"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "text.color": FG,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": AXIS, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "legend.labelcolor": FG,
})
r = json.load(open(RES / "results.json"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


# 1. implied vs the volatility that followed
j = pd.read_parquet(RES / "dvol_vs_rv.parquet")
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.plot(j.index, j.rv * 100, color=SLATE, lw=1.4, label="Realized over the next 30 days")
ax.plot(j.index, j.dvol * 100, color=BLUE, lw=1.6, label="Implied (DVOL) on the day")
ax.set_ylabel("Annualized volatility, %", color=MUTED)
ax.legend(loc="upper right")
ax.set_title("BTC implied volatility against the volatility that followed", loc="left", color=FG, fontsize=13, pad=14)
save(fig, "implied-vs-realized")

# 2. direction: AUC by horizon and feature set (logistic regression)
d = pd.DataFrame(r["direction"])
d = d[d.model == "logit"]
sets = [("market", "Price history", SLATE), ("surface", "Vol surface", BLUE), ("market+surface", "Both", AMBER)]
fig, ax = plt.subplots(figsize=(10, 4.4))
for i, h in enumerate(["4h", "24h", "7d"]):
    for j_, (key, lab, col) in enumerate(sets):
        row = d[(d.horizon == h) & (d.features == key)].iloc[0]
        x = i + (j_ - 1) * 0.22
        ax.plot([x, x], [row.auc_lo, row.auc_hi], color=col, lw=2)
        ax.plot([x], [row.auc], "o", color=col, ms=8, mec=BG, mew=1.5, label=lab if i == 0 else None)
ax.axhline(0.5, color=FG, lw=0.8, alpha=0.6)
ax.set_xticks(range(3), ["Next 4 hours", "Next 24 hours", "Next 7 days"])
ax.set_ylabel("Out-of-sample AUC (0.5 = coin flip)", color=MUTED)
ax.legend(loc="lower left", ncol=3)
ax.set_title("Predicting whether BTC goes up: AUC with 95% intervals", loc="left", color=FG, fontsize=13, pad=14)
save(fig, "direction-auc")

# 3. size of the move: out-of-sample R2 of log realized vol (ridge)
v = pd.DataFrame(r["vol"])
v = v[v.model == "ridge"]
sets = [("HAR", "Past realized vol (HAR)", SLATE), ("implied only", "Implied vol only", BLUE),
        ("HAR+implied", "Both", AMBER), ("HAR+surface", "Both + skew and wings", "#7fb6ff")]
fig, ax = plt.subplots(figsize=(10, 4.4))
w = 0.2
for i, h in enumerate(["24h", "7d"]):
    for j_, (key, lab, col) in enumerate(sets):
        row = v[(v.horizon == h) & (v.features == key)].iloc[0]
        x = i + (j_ - 1.5) * w
        ax.bar(x, row.r2_oos, w * 0.9, color=col, label=lab if i == 0 else None)
        ax.plot([x, x], [row.r2_lo, row.r2_hi], color=FG, lw=1.1)
        ax.annotate(f"{row.r2_oos:.2f}", (x, row.r2_hi), xytext=(0, 4), textcoords="offset points", ha="center",
                    color=FG, fontsize=10)
ax.set_xticks(range(2), ["Realized vol over the next 24 hours", "Realized vol over the next 7 days"])
ax.set_ylabel("Out-of-sample R²", color=MUTED)
ax.set_ylim(0, 0.72)
ax.legend(loc="upper left", ncol=2)
ax.set_title("Predicting how much BTC will move", loc="left", color=FG, fontsize=13, pad=14)
save(fig, "vol-r2")
print("charts in", OUT)
