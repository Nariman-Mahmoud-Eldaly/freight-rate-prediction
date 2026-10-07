"""End-to-end: validate (time-based + unseen-city), fit final model, write all deliverables.
Usage: python train.py"""
import numpy as np, pandas as pd, json
from pathlib import Path
from model import Ensemble, flag_label_outliers
from features import haversine

DATA, OUT = Path("data"), Path("outputs"); OUT.mkdir(exist_ok=True)
train = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
val = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
dec = pd.read_csv(DATA / "december_chart_inputs.csv", parse_dates=["date"])

# market_index is a market-wide daily signal: use the per-day median over ALL feature rows (train+validation)
# to fill missing values and to supply December values for the chart input (which has no market_index column).
daily_mi = pd.concat([train, val]).groupby("date").market_index.median()
train["outlier"] = flag_label_outliers(train, daily_mi)
print(f"rows={len(train)}  corrupted-rate rows removed from training={train.outlier.sum()} ({train.outlier.mean():.2%})")

def metrics(df, pred):
    ok = ~df.outlier.values; y = df.posted_rate.values[ok]; e = np.abs(y - pred[ok])
    return dict(MAE=round(e.mean(), 1), MAPE=round((e / y).mean() * 100, 2), RMSE=round(np.sqrt((e ** 2).mean()), 1))

# ---------- validation ----------
report = {}
# (1) forward-in-time folds: mimic "train on the past, predict the next 2 months" (the real test is Nov-Dec)
for name, (a, b) in {"Jul-Aug": ("2025-07-01", "2025-08-31"), "Sep-Oct": ("2025-09-01", "2025-10-31"),
                     "Oct": ("2025-10-01", "2025-10-31")}.items():
    tr, te = train[train.date < a], train[(train.date >= a) & (train.date <= b)]
    report[f"time_fold_{name}"] = metrics(te, Ensemble(daily_mi).fit(tr).predict(te))
# (2) unseen-city holdout: 8 cities never seen (the validation file has 8 new cities)
rng = np.random.RandomState(0); hold = set(rng.choice(sorted(set(train.pickup)), 8, replace=False))
te = train.pickup.isin(hold) | train.delivery.isin(hold)
report["city_holdout"] = metrics(train[te], Ensemble(daily_mi).fit(train[~te]).predict(train[te]))
report["city_holdout_cities"] = sorted(hold)
for k, v in report.items(): print(k, v)
json.dump(report, open(OUT / "validation_report.json", "w"), indent=2)

# ---------- final fit on ALL labelled data, predict the 12,000 validation loads ----------
final = Ensemble(daily_mi).fit(train)
print("market_index beta (within-day):", round(final.members[2]["beta"], 3))
sub = pd.read_csv(DATA / "validation_predictions_template.csv")[["load_id"]]
val["predicted_rate"] = final.predict(val)
sub = sub.merge(val[["load_id", "predicted_rate"]], on="load_id", how="left")
assert len(sub) == 12000 and sub.predicted_rate.notna().all() and sub.load_id.is_unique
sub["predicted_rate"] = sub.predicted_rate.round(2)
sub.to_csv("validation_predictions.csv", index=False)

# ---------- fixed December chart load ----------
coords = pd.concat([train[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
                    train[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1)]).drop_duplicates("city").set_index("city")
d2 = dec.copy()
d2["pickup_lat"], d2["pickup_lon"] = coords.loc[d2.pickup, "lat"].values, coords.loc[d2.pickup, "lon"].values
d2["delivery_lat"], d2["delivery_lon"] = coords.loc[d2.delivery, "lat"].values, coords.loc[d2.delivery, "lon"].values
d2["market_index"] = np.nan; d2["quote_signal"] = np.nan
dec["predicted_rate"] = final.predict(d2).round(2)
dec["date"] = dec["date"].dt.strftime("%Y-%m-%d")
dec.to_csv("december_chart_inputs.csv", index=False)
print(dec[["date", "predicted_rate"]].describe())
