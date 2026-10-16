"""Side numbers for the write-up: volatility risk premium and the paired gain of implied vol over HAR."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from study import block_boot

ROOT = Path(__file__).parent
DATA, RES = ROOT / "data", ROOT / "results"


def main():
    out = {}
    px = pd.read_parquet(DATA / "btc_1h.parquet")
    close = px.set_index(px["t"] + pd.Timedelta(hours=1))["close"]
    r = np.log(close).diff()
    fwd30 = np.sqrt((r**2).rolling(720).sum().shift(-720) * 365.25 * 24 / 720)
    dv = pd.read_parquet(DATA / "dvol_1h.parquet")
    dvol = dv.set_index(dv["t"] + pd.Timedelta(hours=1))["close"] / 100
    j = pd.concat([dvol.rename("dvol"), fwd30.rename("rv")], axis=1).dropna()
    j = j[j.index.hour == 0]
    out["vrp"] = dict(days=len(j), mean_dvol=j.dvol.mean(), mean_fwd_rv=j.rv.mean(),
                      mean_gap=(j.dvol - j.rv).mean(), share_dvol_above=(j.dvol > j.rv).mean(),
                      corr=np.corrcoef(j.dvol, j.rv)[0, 1])
    j.to_parquet(RES / "dvol_vs_rv.parquet")

    # paired improvement: share of HAR's squared error removed by adding implied vol
    for k, block in [("24h", 42), ("7d", 84)]:
        a = pd.read_parquet(RES / f"vol_{k}_HAR.parquet")
        b = pd.read_parquet(RES / f"vol_{k}_HAR_implied.parquet")
        idx = a.index.intersection(b.index)
        ea = ((a.loc[idx, "y"] - a.loc[idx, "p"]) ** 2).to_numpy()
        eb = ((b.loc[idx, "y"] - b.loc[idx, "p"]) ** 2).to_numpy()
        lo, hi = block_boot(lambda i: 1 - eb[i].sum() / ea[i].sum(), len(idx), block=block)
        out[f"gain_{k}"] = dict(sse_cut=1 - eb.sum() / ea.sum(), lo=lo, hi=hi,
                                months_better=float(((pd.Series(ea, idx) - pd.Series(eb, idx))
                                                     .groupby(idx.to_period("M")).sum() > 0).mean()))
    print(json.dumps(out, indent=1, default=float))
    with open(RES / "extra.json", "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
