"""Does the BTC option surface predict spot? Walk-forward study on a 4-hour grid.

Two questions, same features:
  1. direction: sign of the forward return over 4h, 24h and 7d;
  2. size: log realized volatility over the next 24h and 7d.
Models are retrained every month on all data whose targets were already known
(the target horizon is embargoed). Results go to results/.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA, RES = ROOT / "data", ROOT / "results"
H = {"4h": 4, "24h": 24, "7d": 168}
FIRST_TEST = pd.Timestamp("2022-07-01", tz="UTC")
COST = 0.0005  # per unit of position change, taker fee plus half spread
ANN = np.sqrt(365.25 * 6)

MARKET = ["ret_4h", "ret_24h", "ret_7d", "lrv_1d", "lrv_7d", "lrv_30d"]
SURFACE = ["dvol", "atm7", "atm30", "atm90", "term", "rr7", "rr30", "bf7", "bf30", "vrp", "d_dvol_24h", "d_rr30_24h"]
FLOW = ["flow_call", "flow_put", "vega_flow"]
GROUPS = {"market": MARKET, "surface": SURFACE, "market+surface": MARKET + SURFACE,
          "all": MARKET + SURFACE + FLOW}


def build():
    px = pd.read_parquet(DATA / "btc_1h.parquet")
    # the candle opened at t-1h closes at t: that close is the price known at t
    close = px.set_index(px["t"] + pd.Timedelta(hours=1))["close"]
    r = np.log(close).diff()
    dv = pd.read_parquet(DATA / "dvol_1h.parquet")
    dvol = dv.set_index(dv["t"] + pd.Timedelta(hours=1))["close"] / 100

    s = pd.read_parquet(DATA / "surface.parquet").set_index("t")
    for c in ["atm7", "atm30", "atm90", "rr7", "rr30", "bf7", "bf30"]:
        # thin hours: carry the last fit forward up to 8h, then smooth the noisy wings
        s[c] = s[c].ffill(limit=2)
        if c[:2] in ("rr", "bf"):
            s[c] = s[c].rolling(3, min_periods=1).mean()
    df = s.join(close.rename("close"), how="left").join(dvol.rename("dvol"), how="left")

    rv = lambda n: np.sqrt((r**2).rolling(n).sum() * 365.25 * 24 / n)
    df["ret_4h"] = np.log(close).diff(4).reindex(df.index)
    df["ret_24h"] = np.log(close).diff(24).reindex(df.index)
    df["ret_7d"] = np.log(close).diff(168).reindex(df.index)
    for name, n in [("1d", 24), ("7d", 168), ("30d", 720)]:
        df[f"lrv_{name}"] = np.log(rv(n).reindex(df.index))
    df["term"] = df["atm90"] - df["atm7"]
    df["vrp"] = df["dvol"] - np.exp(df["lrv_30d"])
    df["d_dvol_24h"] = df["dvol"].diff(6)
    df["d_rr30_24h"] = df["rr30"].diff(6)
    for c in FLOW:
        df[c] = np.sign(df[c]) * np.log1p(df[c].abs())

    lc = np.log(close)
    for k, h in H.items():
        df[f"fwd_{k}"] = (lc.shift(-h) - lc).reindex(df.index)
    for k, h in [("24h", 24), ("7d", 168)]:
        fwd_rv = np.sqrt((r**2).rolling(h).sum().shift(-h) * 365.25 * 24 / h)
        df[f"fwd_lrv_{k}"] = np.log(fwd_rv.reindex(df.index))
    return df


def walk_forward(df, feats, target, horizon_h, model_fn, classify):
    """Monthly refits; training rows need their target fully realised before the test month."""
    d = df.dropna(subset=feats + [target])
    preds = pd.Series(np.nan, index=d.index)
    for m in pd.date_range(FIRST_TEST, d.index.max(), freq="MS"):
        test = (d.index >= m) & (d.index < m + pd.offsets.MonthBegin(1))
        train = d.index + pd.Timedelta(hours=horizon_h) < m
        if test.sum() == 0:
            continue
        y = d.loc[train, target]
        model = model_fn().fit(d.loc[train, feats], (y > 0).astype(int) if classify else y)
        X = d.loc[test, feats]
        preds[test] = model.predict_proba(X)[:, 1] if classify else model.predict(X)
    return preds.dropna()


def logit():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))


def gbm_c():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=300, min_samples_leaf=200,
                                          l2_regularization=1.0)


def ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=10))


def gbm_r():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=300, min_samples_leaf=200,
                                         l2_regularization=1.0)


def block_boot(stat, n, block=42, reps=1000, seed=0):
    """Moving-block bootstrap (blocks of 7 days of 4h rows) of a statistic over row indices."""
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    out = []
    for _ in range(reps):
        starts = rng.integers(0, n - block, nb)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        out.append(stat(idx))
    return np.percentile(out, [2.5, 97.5])


def auc_ci(y, p):
    y, p = np.asarray(y), np.asarray(p)
    lo, hi = block_boot(lambda i: roc_auc_score(y[i], p[i]) if y[i].min() != y[i].max() else 0.5, len(y))
    return roc_auc_score(y, p), lo, hi


def strategy(df, p, horizon_h):
    """Long/short on the signal, each signal held for the horizon (overlapping book), 4h bars."""
    sig = np.sign(p - 0.5).reindex(df.index).fillna(0)
    k = max(1, horizon_h // 4)
    pos = sig.rolling(k, min_periods=1).mean()
    r4 = df["fwd_4h"]
    pnl = (pos * r4 - COST * pos.diff().abs().fillna(0)).loc[p.index.min():].dropna()
    return pnl


def sharpe_ci(pnl):
    x = pnl.to_numpy()
    lo, hi = block_boot(lambda i: x[i].mean() / x[i].std() * ANN, len(x))
    return x.mean() / x.std() * ANN, lo, hi


def main():
    RES.mkdir(exist_ok=True)
    df = build()
    df.to_parquet(DATA / "dataset.parquet")
    print("rows", len(df), "with surface", df[SURFACE].notna().all(axis=1).sum())
    out = {"direction": [], "vol": [], "univariate": [], "strategy": []}

    # 1. single features vs forward return: rank correlation with block-bootstrap CI
    d = df.dropna(subset=SURFACE + MARKET + ["fwd_24h", "fwd_7d"])
    d = d[d.index >= FIRST_TEST]
    for f in SURFACE + MARKET:
        for k in ["24h", "7d"]:
            a, b = d[f].rank().to_numpy(), d[f"fwd_{k}"].rank().to_numpy()
            ic = np.corrcoef(a, b)[0, 1]
            lo, hi = block_boot(lambda i: np.corrcoef(a[i], b[i])[0, 1], len(a), block=42 if k == "24h" else 84)
            out["univariate"].append(dict(feature=f, horizon=k, ic=ic, lo=lo, hi=hi))
    print(pd.DataFrame(out["univariate"]).round(3).to_string())

    # 2. direction
    for k, h in H.items():
        for g, feats in GROUPS.items():
            for name, fn in [("logit", logit), ("gbm", gbm_c)]:
                p = walk_forward(df, feats, f"fwd_{k}", h, fn, True)
                y = (df.loc[p.index, f"fwd_{k}"] > 0).astype(int)
                auc, lo, hi = auc_ci(y, p)
                acc = ((p > 0.5).astype(int) == y).mean()
                pnl = strategy(df, p, h)
                sr, slo, shi = sharpe_ci(pnl)
                row = dict(horizon=k, features=g, model=name, n=len(p), auc=auc, auc_lo=lo, auc_hi=hi, hit=acc,
                           base_up=y.mean(), sharpe=sr, sharpe_lo=slo, sharpe_hi=shi)
                out["direction"].append(row)
                print(row, flush=True)
                if k == "24h" and name == "logit":
                    pnl.rename(g).to_frame().to_parquet(RES / f"pnl_24h_{g}.parquet")
    bh = df.loc[df.index >= FIRST_TEST, "fwd_4h"].dropna()
    out["strategy"].append(dict(name="buy_and_hold", sharpe=bh.mean() / bh.std() * ANN))

    # 3. realized volatility
    har = ["lrv_1d", "lrv_7d", "lrv_30d"]
    vol_groups = {"HAR": har, "HAR+implied": har + ["dvol", "atm7", "atm30", "term"],
                  "HAR+surface": har + SURFACE, "implied only": ["dvol", "atm7", "atm30"]}
    for k, h in [("24h", 24), ("7d", 168)]:
        t = f"fwd_lrv_{k}"
        base = df.dropna(subset=MARKET + SURFACE + [t])
        for g, feats in vol_groups.items():
            for name, fn in [("ridge", ridge), ("gbm", gbm_r)]:
                p = walk_forward(base, feats, t, h, fn, False)
                y = base.loc[p.index, t]
                # out-of-sample R2 against the expanding historical mean
                hist = base[t].expanding().mean().shift(h // 4).loc[p.index]
                r2 = 1 - ((y - p) ** 2).sum() / ((y - hist) ** 2).sum()
                yv, pv, hv = y.to_numpy(), p.to_numpy(), hist.to_numpy()
                lo, hi = block_boot(lambda i: 1 - ((yv[i] - pv[i]) ** 2).sum() / ((yv[i] - hv[i]) ** 2).sum(),
                                    len(yv), block=42 if k == "24h" else 84)
                row = dict(horizon=k, features=g, model=name, n=len(p), r2_oos=r2, r2_lo=lo, r2_hi=hi,
                           corr=np.corrcoef(yv, pv)[0, 1])
                out["vol"].append(row)
                print(row, flush=True)
                if name == "ridge":
                    pd.DataFrame({"y": y, "p": p}).to_parquet(RES / f"vol_{k}_{g.replace(' ', '_').replace('+', '_')}.parquet")

    with open(RES / "results.json", "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
