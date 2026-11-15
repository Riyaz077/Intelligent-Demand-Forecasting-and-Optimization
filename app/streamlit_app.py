from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import streamlit as st
import pandas as pd
import plotly.express as px
from src.data import generate_data, validate_data
from src.features import add_time_features, FEATURES
from src.forecast import chronological_split, forecast_group
from src.model import train_models

st.set_page_config(page_title="DemandCast",page_icon="📈",layout="wide")
st.title("📈 DemandCast")
st.caption("Multivariate time-series forecasting for business demand planning | Synthetic demo")
uploaded=st.sidebar.file_uploader("Upload demand history CSV",type=["csv"])
if uploaded:
    df=validate_data(pd.read_csv(uploaded))
else:
    path=Path(__file__).resolve().parents[1]/"data"/"demand_history.csv"
    df=validate_data(pd.read_csv(path)) if path.exists() else generate_data()
stores=sorted(df.store_id.unique())
products=sorted(df.product_id.unique())
store=st.sidebar.selectbox("Store",stores)
product=st.sidebar.selectbox("Product",products)
group=df[(df.store_id==store)&(df.product_id==product)].sort_values("date")
c1,c2,c3=st.columns(3)
c1.metric("History rows",f"{len(df):,}")
c2.metric("Stores",len(stores))
c3.metric("Products",len(products))
st.subheader("Historical demand")
st.plotly_chart(px.line(group,x="date",y="demand",title=f"{store} · {product}"),use_container_width=True)
st.subheader("Train and compare forecasting models")
if st.button("Train models and forecast next 14 days",type="primary"):
    with st.spinner("Engineering features and training models..."):
        engineered=add_time_features(df)
        tr,te=chronological_split(engineered,60)
        fitted,metrics,best=train_models(tr[FEATURES],tr.demand,te[FEATURES],te.demand)
        model=fitted[best]
        hist=group.copy()
        forecast=forecast_group(hist,model,14)
    st.success(f"Best model by holdout RMSE: {best}")
    st.dataframe(pd.DataFrame(metrics).T,use_container_width=True)
    st.plotly_chart(px.line(forecast,x="date",y="predicted_demand",
        title="14-day recursive demand forecast"),use_container_width=True)
    st.dataframe(forecast,use_container_width=True)
st.info("Synthetic data and illustrative forecasts. Validate against real operational data before planning inventory or capacity.")
