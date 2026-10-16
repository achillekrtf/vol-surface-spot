"""Machine learning version of the daily straddle filter.

Instead of a fixed rule (sell when IV > HAR forecast + k), a model predicts the P&L of tomorrow's
straddle from the surface, past realized volatility, returns and option flow at 08:00, and the
straddle is sold when the predicted P&L is positive. Models are refitted every month on straddles
that had already expired, starting from April 2021, and tested from July 2022 like the rule.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from study import FIRST_TEST, FLOW, MARKET, SURFACE
from vol_selling import stats

ROOT = Path(__file__).parent
DATA, RES = ROOT / "data", ROOT / "results"
SPREAD = ["iv", "iv_rv1", "iv_rv7", "iv_rv30", "iv_dvol"]
SETS = {"spread only": SPREAD, "surface + market": SPREAD + MARKET + SURFACE,
        "everything": SPREAD + MARKET + SURFACE + FLOW}


def models():
    return {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=30)),
        "gbm": lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=300,
                                                     min_samples_leaf=60, l2_regularization=1.0),
        "gbm, P(win)": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=300,
                                                             min_samples_leaf=60, l2_regularization=1.0),
        "logit, P(win)": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)),
    }


def build():
    tr = pd.read_parquet(RES / "straddles.parquet")
    tr = tr[tr["kind"] == "daily"]
    ds = pd.read_parquet(DATA / "dataset.parquet")
    df = tr.join(ds[MARKET + SURFACE + FLOW], how="left")
    df["iv_rv1"] = df["iv"] - np.exp(df["lrv_1d"])
    df["iv_rv7"] = df["iv"] - np.exp(df["lrv_7d"])
    df["iv_rv30"] = df["iv"] - np.exp(df["lrv_30d"])
    df["iv_dvol"] = df["iv"] - df["dvol"]
    return df


def walk_forward(df, feats, target, make, classify):
    d = df.dropna(subset=feats + [target])
    preds = pd.Series(np.nan, index=d.index)
    for m in pd.date_range(FIRST_TEST, d.index.max(), freq="MS"):
        test = (d.index >= m) & (d.index < m + pd.offsets.MonthBegin(1))
        train = d.index + pd.Timedelta(days=1) <= m  # expired before the test month starts
        if test.sum() == 0:
            continue
        y = d.loc[train, target]
        model = make().fit(d.loc[train, feats], (y > 0).astype(int) if classify else y)
        X = d.loc[test, feats]
        preds[test] = model.predict_proba(X)[:, 1] if classify else model.predict(X)
    return preds.dropna()


def main():
    df = build()
    test = df[(df.index >= FIRST_TEST) & df["har"].notna()]
    out = []
    for c in ["pnl_0.01", "pnl_0.03"]:
        for k in [None, 0.05, 0.10]:
            sel = test if k is None else test[test["iv"] > test["har"] + k]
            s = stats(sel[c], 365.25)
            out.append(dict(rule="always" if k is None else f"IV > HAR + {int(k * 100)}", cost=c, **s))
    all_preds = {}
    for fs, feats in SETS.items():
        for name, make in models().items():
            classify = "P(win)" in name
            # trained on the 1-point P&L; the 3-point cost is applied to the same decisions
            p = walk_forward(df, feats, "pnl_0.01", make, classify)
            p = p[p.index.isin(test.index)]
            # classifiers: sell when the win probability beats the base rate seen in training
            base = (df.loc[df.index < FIRST_TEST, "pnl_0.01"] > 0).mean() if classify else 0.0
            sell = p > base
            all_preds[f"{fs} | {name}"] = sell
            for c in ["pnl_0.01", "pnl_0.03"]:
                sel = test.loc[sell[sell].index, c]
                s = stats(sel, 365.25)
                if s:
                    out.append(dict(rule=f"ML: {name}", features=fs, cost=c, **s,
                                    days=int(sell.sum())))
    res = pd.DataFrame(out)
    print(res[["rule", "features", "cost", "n", "mean", "lo", "hi", "win", "worst", "total", "max_dd"]]
          .round(4).to_string())
    pd.DataFrame(all_preds).to_parquet(RES / "ml_selling_decisions.parquet")
    with open(RES / "ml_selling.json", "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
