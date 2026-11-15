from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.data import generate_data, validate_data
from src.features import add_time_features, FEATURES
from src.forecast import chronological_split
from src.model import train_models

path=Path("data/demand_history.csv")
df=validate_data(pd.read_csv(path)) if path.exists() else generate_data()
engineered=add_time_features(df)
train,test=chronological_split(engineered,60)
fitted,metrics,best=train_models(train[FEATURES],train.demand,test[FEATURES],test.demand)
print("Best model:",best)
print("Metrics:",metrics)
