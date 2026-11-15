import numpy as np
import pandas as pd

def generate_data(days=900, stores=5, products=12, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    rows = []
    for s in range(1, stores+1):
        for p in range(1, products+1):
            category = ["Grocery","Home","Electronics","Personal Care"][(p-1)%4]
            base = rng.uniform(18, 95)
            phase = rng.uniform(0, 2*np.pi)
            price_base = rng.uniform(5, 120)
            for i, date in enumerate(dates):
                weekly = 1 + 0.16*np.sin(2*np.pi*i/7 + phase)
                yearly = 1 + 0.13*np.sin(2*np.pi*i/365.25 + phase/2)
                trend = 1 + 0.00025*i
                promotion = int(rng.random() < 0.13)
                holiday = int(date.day in [1, 15, 25] and date.month in [1, 8, 10, 12])
                temperature = 25 + 7*np.sin(2*np.pi*i/365.25) + rng.normal(0, 2)
                price = max(1, price_base*(1 - 0.08*promotion) + rng.normal(0, price_base*0.025))
                mean = base*weekly*yearly*trend*(1+0.28*promotion)*(1+0.22*holiday)
                mean *= np.exp(-0.35*((price/price_base)-1))
                mean *= (1 + 0.003*(temperature-25))
                demand = max(0, int(rng.poisson(max(mean, 0.2))))
                rows.append((date, f"STORE{s:02d}", f"PROD{p:03d}", category,
                             round(price,2), promotion, round(temperature,2), holiday, demand))
    return pd.DataFrame(rows, columns=["date","store_id","product_id","category","price",
                                       "promotion","temperature","holiday","demand"])

REQUIRED = ["date","store_id","product_id","category","price","promotion","temperature","holiday","demand"]

def validate_data(df):
    missing = sorted(set(REQUIRED)-set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in ["price","promotion","temperature","holiday","demand"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=REQUIRED)
    out = out[(out.price > 0) & (out.demand >= 0)]
    return out.sort_values(["store_id","product_id","date"]).reset_index(drop=True)
