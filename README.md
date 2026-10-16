# Can the Bitcoin options surface predict spot?

Code for the article on [achillekrtf.github.io](https://achillekrtf.github.io/articles/btc-vol-surface-ml/).

Every 4 hours from April 2021 to October 2026, the BTC implied volatility surface is rebuilt from the Deribit option
trades of the previous hour (about 12,000 snapshots, 5 million trades). Its features are then used to predict, out of
sample with monthly refits:

- the direction of BTC over the next 4 hours, 24 hours and 7 days (logistic regression, gradient boosting);
- realized volatility over the next 24 hours and 7 days, against the HAR benchmark.

Headline: direction is a coin flip (AUC 0.51 to 0.53, below 0.5 at 7 days), while implied volatility removes about 9%
of HAR's squared error on realized volatility.

## Run

```bash
pip install pandas numpy scipy scikit-learn pyarrow matplotlib requests
python fetch_spot.py       # Binance 1h candles and Deribit DVOL
python fetch_options.py    # Deribit option trades around each snapshot (data/options/, under an hour)
python surface.py          # smile fits -> data/surface.parquet
python study.py            # walk-forward models -> results/results.json
python extra.py            # volatility risk premium, paired gain over HAR
python charts.py           # charts/
```

| File | What it does |
| --- | --- |
| `fetch_options.py` | Trades in `[t - 1h, t)` for every 4-hour snapshot, from Deribit's history API |
| `surface.py` | Quadratic smile in delta per expiry, ATM / 25-delta risk reversal / butterfly, interpolated to 7, 30, 90 days |
| `study.py` | Features, targets, monthly walk-forward with the target horizon embargoed, block-bootstrap intervals |
| `extra.py` | Implied vs realized volatility, share of HAR error removed by implied vol |

Personal research, not investment advice.
