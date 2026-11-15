import numpy as np
import pandas as pd

FEATURES = ["store_id","product_id","category","price","promotion","temperature","holiday",
            "day_of_week","day_of_month","month","day_of_year","lag_1","lag_7","lag_14",
            "rolling_mean_7","rolling_std_7","rolling_mean_28"]

def add_time_features(df):
    parts = []
    for _, group in df.groupby(["store_id","product_id"], sort=False):
        g = group.sort_values("date").copy()
        g["day_of_week"] = g.date.dt.dayofweek
        g["day_of_month"] = g.date.dt.day
        g["month"] = g.date.dt.month
        g["day_of_year"] = g.date.dt.dayofyear
        shifted = g["demand"].shift(1)
        g["lag_1"] = shifted
        g["lag_7"] = g["demand"].shift(7)
        g["lag_14"] = g["demand"].shift(14)
        g["rolling_mean_7"] = shifted.rolling(7).mean()
        g["rolling_std_7"] = shifted.rolling(7).std()
        g["rolling_mean_28"] = shifted.rolling(28).mean()
        parts.append(g)
    return pd.concat(parts, ignore_index=True).dropna(subset=FEATURES[7:]).reset_index(drop=True)

def make_xy(df):
    from src.features import FEATURES
    engineered = add_time_features(df)
    return engineered[FEATURES], engineered["demand"].astype(float), engineered
