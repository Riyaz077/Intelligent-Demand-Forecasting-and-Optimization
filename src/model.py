from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.features import FEATURES

CAT = ["store_id","product_id","category"]
NUM = [c for c in FEATURES if c not in CAT]

def build_preprocessor():
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("numeric", StandardScaler(), NUM)
    ])

def evaluate(y_true, y_pred):
    return {"mae":float(mean_absolute_error(y_true,y_pred)),
            "rmse":float(np.sqrt(mean_squared_error(y_true,y_pred))),
            "r2":float(r2_score(y_true,y_pred))}

def train_models(X_train, y_train, X_test, y_test, artifact_dir="artifacts", report_dir="reports"):
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=160, max_depth=16,
            min_samples_leaf=2, random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=180,
            learning_rate=0.08, l2_regularization=1.0, random_state=42)
    }
    results, fitted = {}, {}
    for name, estimator in models.items():
        pipe = Pipeline([("prep",build_preprocessor()),("model",estimator)])
        pipe.fit(X_train,y_train)
        pred = np.maximum(pipe.predict(X_test),0)
        results[name] = evaluate(y_test,pred)
        fitted[name] = pipe
    baseline = np.maximum(X_test["lag_7"].to_numpy(),0)
    results["Seasonal Naive (lag 7)"] = evaluate(y_test,baseline)
    best_name = min([k for k in results if k != "Seasonal Naive (lag 7)"],
                    key=lambda k:results[k]["rmse"])
    Path(artifact_dir).mkdir(parents=True,exist_ok=True)
    Path(report_dir).mkdir(parents=True,exist_ok=True)
    joblib.dump(fitted[best_name],Path(artifact_dir)/"best_model.joblib")
    joblib.dump(fitted,Path(artifact_dir)/"all_models.joblib")
    (Path(report_dir)/"metrics.json").write_text(json.dumps(results,indent=2))
    (Path(artifact_dir)/"model_info.json").write_text(json.dumps({"best_model":best_name},indent=2))
    return fitted, results, best_name
