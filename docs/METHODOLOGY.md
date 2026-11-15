# Methodology

1. Generate or load daily demand observations for each store-product series.
2. Validate fields, numeric ranges and dates.
3. Create calendar features, lagged demand and rolling statistics using only prior observations.
4. Split chronologically, reserving the most recent period for evaluation.
5. Train Random Forest and HistGradientBoosting regressors and compare with a seasonal-naive lag-7 baseline.
6. Report MAE, RMSE and R².
7. Generate recursive forecasts using lagged predictions for future steps.

## Leakage note
Feature lags and rolling statistics are shifted so the current target is not included in its own features. The split is time-ordered rather than randomly shuffled.
