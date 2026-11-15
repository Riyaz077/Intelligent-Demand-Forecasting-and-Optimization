import pandas as pd
import numpy as np
from src.features import add_time_features, FEATURES

def chronological_split(engineered, test_days=60):
    cutoff = engineered["date"].max() - pd.Timedelta(days=test_days-1)
    train = engineered[engineered.date < cutoff]
    test = engineered[engineered.date >= cutoff]
    if train.empty or test.empty:
        raise ValueError("Not enough history for requested chronological split.")
    return train, test

def forecast_group(history, model, horizon=14):
    hist = history.sort_values("date").copy()
    if len(hist) < 35:
        raise ValueError("At least 35 historical observations are required.")
    outputs = []
    for _ in range(horizon):
        last = hist.iloc[-1]
        next_date = last.date + pd.Timedelta(days=1)
        recent = hist.demand.tail(28)
        row = {
            "date":next_date, "store_id":last.store_id, "product_id":last.product_id,
            "category":last.category, "price":float(hist.price.tail(7).mean()),
            "promotion":0, "temperature":float(hist.temperature.tail(7).mean()),
            "holiday":0, "demand":np.nan,
            "day_of_week":next_date.dayofweek, "day_of_month":next_date.day,
            "month":next_date.month, "day_of_year":next_date.dayofyear,
            "lag_1":float(hist.demand.iloc[-1]), "lag_7":float(hist.demand.iloc[-7]),
            "lag_14":float(hist.demand.iloc[-14]), "rolling_mean_7":float(recent.tail(7).mean()),
            "rolling_std_7":float(recent.tail(7).std(ddof=1) or 0),
            "rolling_mean_28":float(recent.mean())
        }
        x = pd.DataFrame([{k:row[k] for k in FEATURES}])
        prediction = max(0.0,float(model.predict(x)[0]))
        row["demand"] = prediction
        outputs.append({"date":next_date,"predicted_demand":prediction,
                        "store_id":last.store_id,"product_id":last.product_id})
        hist = pd.concat([hist,pd.DataFrame([row])],ignore_index=True)
    return pd.DataFrame(outputs)
