# DemandCast
## Multivariate Time-Series Forecasting for Business Demand Planning

DemandCast is an end-to-end forecasting portfolio project for business demand planning. It combines historical demand with calendar, price, promotion, weather and product context to forecast future demand, compare forecasting models, and support inventory and capacity planning.

### Features
- Reproducible synthetic multivariate daily demand dataset for multiple products and stores.
- Time-aware feature engineering: lags, rolling statistics, calendar signals and external drivers.
- Chronological train/test split to reduce time-series leakage.
- Baseline, Random Forest and HistGradientBoosting forecasting models.
- MAE, RMSE and R² evaluation.
- Recursive multi-step forecast generation.
- Streamlit dashboard for product/store selection, forecast visualization and model comparison.
- FastAPI forecast endpoint.
- Notebook, documentation and tests.

### Quick start
Python 3.10+ recommended.

```bash
pip install -r requirements.txt
python scripts/generate_data.py
python scripts/train.py
streamlit run app/streamlit_app.py
```

Start the API:
```bash
uvicorn api.main:app --reload
```

### Dataset fields
`date`, `store_id`, `product_id`, `category`, `price`, `promotion`, `temperature`, `holiday`, `demand`.

### Project structure
```text
DemandCast/
├── api/              # FastAPI forecasting endpoint
├── app/              # Streamlit dashboard
├── data/             # Synthetic multivariate time-series dataset
├── docs/             # Data dictionary and methodology
├── notebooks/        # Exploratory walkthrough
├── scripts/          # Data generation and model training
├── src/              # Feature engineering and forecasting pipeline
├── tests/            # Unit tests
├── requirements.txt
└── README.md
```

### Important limitations
This is an educational portfolio prototype using synthetic data. The external drivers are simulated and do not represent actual weather, holidays, prices or market conditions. Forecasts are not validated for operational use. Real deployments require reliable business data, backtesting across multiple time windows, monitoring, uncertainty estimates and human review.
