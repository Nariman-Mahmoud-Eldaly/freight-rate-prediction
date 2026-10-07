import numpy as np, pandas as pd

EQUIP = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}

def haversine(lat1, lon1, lat2, lon2):
    a, b, c, e = map(np.radians, [lat1, lon1, lat2, lon2])
    x = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((e - b) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(x))

def build_features(df, daily_mi=None, daily_qs=None, use_market=True, use_quote=True):
    """Row-level feature builder. Uses NO city identity (8 validation cities never appear in training)."""
    f = pd.DataFrame(index=df.index)
    dist = df["distance"].astype(float)
    f["log_dist"] = np.log(dist)
    f["distance"] = dist
    f["equip"] = df["equipment"].map(EQUIP)
    # weight: negative = sign error -> abs; 47,500 = capped; NaN stays NaN (LightGBM handles it)
    w = df["weight"].abs()
    f["weight"] = w
    f["weight_missing"] = w.isna().astype(int)
    f["weight_capped"] = (w >= 47500).astype(int)
    # geography (generalises to new cities, unlike city IDs)
    f["p_lat"], f["p_lon"] = df["pickup_lat"], df["pickup_lon"]
    f["d_lat"], f["d_lon"] = df["delivery_lat"], df["delivery_lon"]
    hv = haversine(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    f["circuity"] = dist / hv.clip(lower=1)
    f["dlat"] = df.delivery_lat - df.pickup_lat
    f["dlon"] = df.delivery_lon - df.pickup_lon
    # calendar (no month / ordinal date: train ends Oct, test is Nov-Dec -> trees can't extrapolate)
    f["dow"] = df["date"].dt.dayofweek
    f["dom"] = df["date"].dt.day
    if use_market:
        mi = df["market_index"]
        if daily_mi is not None:  # impute by that day's market-wide median
            mi = mi.fillna(df["date"].map(daily_mi))
        f["market_index"] = mi
        f["mi_missing"] = df["market_index"].isna().astype(int)
    if use_quote:
        f["quote_signal"] = df["quote_signal"]
    return f
