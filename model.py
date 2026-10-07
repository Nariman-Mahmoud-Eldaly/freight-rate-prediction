"""Cleaning + 3-model LightGBM ensemble for freight rate prediction."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import HuberRegressor
from features import build_features

LGB_PARAMS = dict(objective="huber", alpha=0.3, learning_rate=0.03, num_leaves=15,
                  min_child_samples=60, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=5, n_estimators=700, verbose=-1)

def flag_label_outliers(df, daily_mi):
    """~1.4% of posted_rate values are corrupted (x~4.5 or /~4.5). A robust fit shows a clean gap:
    |log residual| is < 0.3 for 98.6% of rows and > 0.5 for the rest -> drop those rows from training."""
    f = build_features(df, daily_mi).fillna(0)[["log_dist", "equip", "market_index", "quote_signal"]]
    y = np.log(df["posted_rate"])
    resid = y - HuberRegressor(max_iter=500).fit(f, y).predict(f)
    return (np.abs(resid) > 0.5).values

def within_day_beta(tr, mask):
    """Effect of market_index on log(rate), estimated ONLY from within-day variation
    (day fixed effects), because day-level market_index is confounded with a slow quarterly level drift."""
    g = tr[mask & tr["market_index"].notna().values]
    yd = g.lr - g.groupby("date").lr.transform("mean")
    xd = g.market_index - g.groupby("date").market_index.transform("mean")
    Z = np.c_[np.log(g.distance), np.log(g.distance) ** 2, pd.get_dummies(g.equipment).astype(float).values]
    Z = Z - pd.DataFrame(Z, index=g.index).groupby(g.date).transform("mean").values
    return np.linalg.lstsq(np.c_[xd, Z], yd.values, rcond=None)[0][0]

def _fit_one(tr, ref_date, daily_mi, use_mi, half_life):
    clean = ~tr["outlier"].values
    X = build_features(tr, daily_mi, use_market=False, use_quote=False)
    beta = within_day_beta(tr, clean) if use_mi else 0.0
    mi = tr.market_index.fillna(tr.date.map(daily_mi)).values
    w = np.ones(len(tr))
    if half_life:  # recent loads count more -> model anchors to the latest price level
        w = 0.5 ** ((ref_date - tr.date).dt.days.values / half_life)
    g = lgb.LGBMRegressor(**LGB_PARAMS).fit(X[clean], (tr.lr.values - beta * mi)[clean], sample_weight=w[clean])
    return dict(gbm=g, beta=beta)

# (use_market_index, recency half-life in days)
VARIANTS = {"no_mi_equal": (False, None), "no_mi_recent45": (False, 45), "mi_recent30": (True, 30)}

class Ensemble:
    def __init__(self, daily_mi): self.daily_mi = daily_mi
    def fit(self, tr):
        tr = tr.copy(); tr["lr"] = np.log(tr.posted_rate); ref = tr.date.max()
        self.members = [_fit_one(tr, ref, self.daily_mi, mi, hl) for mi, hl in VARIANTS.values()]
        return self
    def predict(self, df):
        X = build_features(df, self.daily_mi, use_market=False, use_quote=False)
        mi = df.market_index.fillna(df.date.map(self.daily_mi)).values
        logp = [m["gbm"].predict(X) + m["beta"] * mi for m in self.members]
        return np.exp(np.mean(logp, axis=0))   # geometric mean of the 3 members
