from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from src.data import generate_data
from src.features import add_time_features, FEATURES
from src.forecast import chronological_split, forecast_group
from src.model import train_models

app=FastAPI(title="DemandCast API",version="1.0.0")
class ForecastRequest(BaseModel):
    store_id:str
    product_id:str
    horizon:int=Field(default=14,ge=1,le=60)

@app.get("/health")
def health():
    return {"status":"ok","project":"DemandCast"}

@app.post("/forecast")
def forecast(req:ForecastRequest):
    try:
        df=generate_data(500,stores=5,products=12,seed=42)
        group=df[(df.store_id==req.store_id)&(df.product_id==req.product_id)].sort_values("date")
        if group.empty:
            raise HTTPException(status_code=404,detail="Store/product combination not found in demo data.")
        engineered=add_time_features(df)
        train,test=chronological_split(engineered,45)
        fitted,metrics,best=train_models(train[FEATURES],train.demand,test[FEATURES],test.demand,
            artifact_dir="/tmp/demandcast_artifacts",report_dir="/tmp/demandcast_reports")
        result=forecast_group(group,fitted[best],req.horizon)
        return {"model":best,"metrics":metrics,"forecast":result.to_dict(orient="records"),
                "note":"Forecast based on synthetic demo history."}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500,detail=str(exc))
