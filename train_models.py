"""
Trains and saves the exact winning model pipeline for each of the three
malaria targets, using the identical procedure (same features, same 80/20
split with random_state=42, same 5-fold GridSearchCV hyperparameter grids)
reported in Chapter 4 of the thesis.

This script does NOT change the methodology in any way. It reruns the same
GridSearchCV already performed in analysis/train_evaluate.py, but only for
the three (target, model) combinations that were identified in Chapter 4
as the best performer for that target, and saves the resulting fitted
scikit-learn Pipeline (SimpleImputer -> MinMaxScaler -> model) to disk so
the Streamlit app can load it without retraining.

After saving, it re-predicts the held-out test set and prints the metrics
side by side with the values reported in Chapter 4 Table 4.5, as a
self-check that nothing has drifted.

Run this once before starting the app:
    python3 train_models.py
"""
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "analytic_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

FEATURES = [
    "Fevers U5",
    "Attendance U5 Public",
    "Attendance U5 Private",
    "Attendance O5 Private",
    "Tested U5 Public",
    "Tested U5 Private",
    "Untested Received AM Public",
    "Tested Negative Received AM Public",
    "TPR",
]

# Winning (target -> model) combinations, as reported in Chapter 4 Table 4.6.
# These were NOT re-selected here; they are the models already identified as
# best-performing in Chapter 4 and are simply refit here so they can be saved.
WINNERS = {
    "Incidence Rate": "XGBoost",
    "Infection Prevalence": "SVR",
    "Mortality Rate": "XGBoost",
}

# Chapter 4 Table 4.5 reference values, used only to self-check this rerun.
REPORTED = {
    ("Incidence Rate", "XGBoost"): dict(RMSE=35.268, MAE=25.313, R2=-0.699),
    ("Infection Prevalence", "SVR"): dict(RMSE=3.664, MAE=3.071, R2=-0.344),
    ("Mortality Rate", "XGBoost"): dict(RMSE=7.949, MAE=7.013, R2=0.826),
}

MODEL_SPECS = {
    "XGBoost": {
        "estimator": XGBRegressor(
            objective="reg:squarederror", subsample=0.8, random_state=RANDOM_STATE
        ),
        "param_grid": {
            "model__n_estimators": [100, 200, 500],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 5, 7],
        },
    },
    "SVR": {
        "estimator": SVR(kernel="rbf"),
        "param_grid": {
            "model__C": [0.1, 1, 10, 100],
            "model__gamma": ["scale", "auto"],
            "model__epsilon": [0.01, 0.1, 0.5],
        },
    },
}

MODEL_SLUG = {"XGBoost": "xgboost", "SVR": "svr", "Random Forest": "random_forest"}
TARGET_SLUG = {
    "Incidence Rate": "incidence",
    "Infection Prevalence": "prevalence",
    "Mortality Rate": "mortality",
}


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH.relative_to(BASE_DIR)}: {df.shape[0]} rows, "
          f"{len(FEATURES)} predictors, years {df['Year'].min()}-{df['Year'].max()}")

    saved_paths = {}
    check_rows = []

    for target, model_name in WINNERS.items():
        # Keep X as a DataFrame (not .values) so the fitted Pipeline retains
        # feature names, matching how app.py will later call .predict() with
        # a named DataFrame and avoiding a spurious sklearn name-mismatch
        # warning at inference time.
        X = df[FEATURES]
        y = df[target].values

        X_train, X_test, y_train, y_test, years_train, years_test = train_test_split(
            X, y, df["Year"].values, test_size=0.2, random_state=RANDOM_STATE
        )

        spec = MODEL_SPECS[model_name]
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", MinMaxScaler()),
            ("model", spec["estimator"]),
        ])
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        grid = GridSearchCV(pipe, spec["param_grid"], cv=cv, scoring="r2", n_jobs=-1)
        grid.fit(X_train, y_train)

        best_pipeline = grid.best_estimator_
        y_pred = best_pipeline.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        ref = REPORTED[(target, model_name)]
        check_rows.append({
            "Target": target, "Model": model_name,
            "RMSE_now": round(rmse, 3), "RMSE_ch4": ref["RMSE"],
            "MAE_now": round(mae, 3), "MAE_ch4": ref["MAE"],
            "R2_now": round(r2, 3), "R2_ch4": ref["R2"],
            "Match": abs(rmse - ref["RMSE"]) < 0.01 and abs(mae - ref["MAE"]) < 0.01 and abs(r2 - ref["R2"]) < 0.01,
        })

        filename = f"{TARGET_SLUG[target]}_{MODEL_SLUG[model_name]}.pkl"
        path = MODELS_DIR / filename
        # joblib.dump writes a pickle. Safe here: this file is written and
        # later read only within this project (by app.py, same machine/repo),
        # never accepted as input from an external or untrusted source.
        joblib.dump({
            "pipeline": best_pipeline,
            "features": FEATURES,
            "target": target,
            "model_name": model_name,
            "test_rmse": rmse,
            "test_mae": mae,
            "test_r2": r2,
            "train_years": sorted(int(y) for y in years_train),
            "test_years": sorted(int(y) for y in years_test),
        }, path)
        saved_paths[target] = path
        print(f"Saved {path.relative_to(BASE_DIR)}  (best_params={grid.best_params_})")

    print("\n=== Self-check against Chapter 4 Table 4.5 ===")
    check_df = pd.DataFrame(check_rows)
    print(check_df.to_string(index=False))
    if not check_df["Match"].all():
        print("\nWARNING: one or more refit metrics do not match Chapter 4 within "
              "tolerance. Do not ship these model files until this is resolved.")
    else:
        print("\nAll refit metrics match Chapter 4 Table 4.5 within tolerance. "
              "Saved pipelines are faithful to the reported results.")

    with open(MODELS_DIR / "manifest.json", "w") as f:
        json.dump({t: str(p.name) for t, p in saved_paths.items()}, f, indent=2)


if __name__ == "__main__":
    main()
