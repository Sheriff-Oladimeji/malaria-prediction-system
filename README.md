# Pediatric Malaria Prediction System

An interactive Streamlit application that exposes the machine-learning models
developed and reported in Chapter 4 of the thesis *"Development of a Machine
Learning-Based Predictive Model for Pediatric Malaria Prevalence in
Nigeria."* It lets a user enter healthcare service-utilisation indicators and
receive a prediction for one of three malaria outcome targets.

This application does not train or change the methodology in any way. It
loads the exact fitted scikit-learn pipelines produced by `train_models.py`,
which reruns the identical Chapter 4 procedure for the three (target, model)
combinations already identified there as best-performing, and simply serves
predictions from them.

## Project Overview

The underlying study trains and compares three regression algorithms
(Random Forest, XGBoost, Support Vector Regression) to predict three
paediatric malaria outcomes for Nigeria from national malaria service
utilisation indicators. Chapter 4 found that no single algorithm wins on
every target: XGBoost performed best for Incidence Rate and Mortality Rate,
and SVR performed best for Infection Prevalence. Only the Mortality Rate
model achieved a positive R² on the held-out test set; this app surfaces
that distinction explicitly rather than hiding it.

## Dataset Description

`data/analytic_dataset.csv` is the final analytic dataset produced by the
Chapter 4 preprocessing pipeline: 15 yearly observations for Nigeria
(2010-2024), merged from the Vector Atlas "National Unit" epidemiological
dataset and "Scenario Estimates" service-utilisation dataset.

## Features Used (9, in this exact order)

1. Fevers U5
2. Attendance U5 Public
3. Attendance U5 Private
4. Attendance O5 Private
5. Tested U5 Public
6. Tested U5 Private
7. Untested Received AM Public
8. Tested Negative Received AM Public
9. TPR (Test Positivity Rate, derived: (Tested Positive Public + Tested
   Positive Private) / (Tested U5 Public + Tested U5 Private))

Note: `ACT_Fever_Ratio` was also engineered in Chapter 4 but was removed by
the |r| > 0.95 Pearson correlation filter (it correlated 0.9835 with
`Tested U5 Public`), so it is not one of the 9 features and is not present
anywhere in this application.

## Target Variables

- **Mortality Rate** (deaths per 100,000) — best model: XGBoost, Test R² = 0.826
- **Incidence Rate** (cases per 1,000) — best model: XGBoost, Test R² = -0.699
- **Infection Prevalence** (% of children under 5) — best model: SVR, Test R² = -0.344

## Machine-Learning Models

Each target has one dedicated saved pipeline, matching Chapter 4 Table 4.6:

| Target | Model | File |
|---|---|---|
| Mortality Rate | XGBoost | `models/mortality_xgboost.pkl` |
| Incidence Rate | XGBoost | `models/incidence_xgboost.pkl` |
| Infection Prevalence | SVR | `models/prevalence_svr.pkl` |

## Preprocessing Pipeline

Each saved file is a single scikit-learn `Pipeline`:

```
Input features -> SimpleImputer(strategy="mean") -> MinMaxScaler() -> Model
```

The imputer and scaler were fit only on the training partition during
`train_models.py`, exactly as required by Chapter 4 Section 3.7.4 (leakage
prevention). The Streamlit app never re-fits or re-scales anything; it only
calls `.predict()` on the already-fitted pipeline.

## Model Evaluation

Full held-out test-set results (Chapter 4, Table 4.5), also shown live in the
app's "Model Performance" section:

| Target | Model | RMSE | MAE | Test R² |
|---|---|---|---|---|
| Incidence Rate | XGBoost | 35.268 | 25.313 | -0.699 |
| Infection Prevalence | SVR | 3.664 | 3.071 | -0.344 |
| Mortality Rate | XGBoost | 7.949 | 7.013 | 0.826 |

Both Incidence Rate and Infection Prevalence have negative test R², meaning
neither model outperforms a naive mean-baseline on the 3 held-out test years
(2010, 2019, 2021). The app displays this fact directly to the user rather
than presenting all three targets as equally reliable.

## Installation

```bash
cd malaria_prediction_system
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Training the Models (run once, before first use)

```bash
python3 train_models.py
```

This reruns the same GridSearchCV procedure used in Chapter 4 for the three
winning (target, model) pairs only, saves the fitted pipelines to `models/`,
and prints a self-check comparing the resulting RMSE/MAE/R² against the
values reported in Chapter 4 Table 4.5 (they should match to 3 decimal
places; if they do not, do not use the application until the discrepancy is
resolved).

## Running the Application

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`) in a
browser.

## How the Prediction Process Works

1. The user enters values for the 9 predictor variables (or toggles
   "Calculate TPR automatically" to derive TPR from Tested Positive
   Public/Private counts instead of entering it directly).
2. The user selects a prediction target (Mortality Rate, Incidence Rate, or
   Infection Prevalence). Mortality Rate is the default, since it is the
   only target with a positive test R² in Chapter 4.
3. On clicking **Predict**, the app assembles the 9 values into a
   single-row `pandas.DataFrame` with columns in the exact order the model
   was trained on, loads the corresponding saved `Pipeline` from `models/`,
   and calls `.predict()`. No training or scaling logic runs in the app
   itself; both steps are frozen inside the loaded pipeline.
4. The prediction, the model name, and its Chapter 4 test R² are displayed
   together. If R² is negative, a warning explains that the model performed
   worse than a mean-baseline on the held-out years, so the number should
   be read as illustrative of the deployed pipeline rather than as a
   trustworthy forecast.

## Testing

`train_models.py` includes a built-in self-check (see above). In addition,
the following was manually verified against Chapter 4 Table 4.7:

Entering the actual 2010 row (Fevers U5=1.42e8, Attendance U5 Public=35665440,
Attendance U5 Private=72240476, Attendance O5 Private=98148341,
Tested U5 Public=2834856.91217, Tested U5 Private=2683779,
Untested Received AM Public=24607789, Tested Negative Received AM
Public=1003239, TPR=1.0331694885) with the Mortality Rate target reproduces
the exact prediction reported in Chapter 4 Table 4.7 (126.89, rounding).

Also verified: all-zero inputs and extremely large (1e15) inputs both return
a numeric prediction without the app crashing (the pipeline extrapolates
rather than erroring); `st.number_input` widgets prevent non-numeric or
missing input at the UI layer, so those failure modes cannot occur through
the form itself. Inputs far outside the observed 2010-2024 range trigger a
non-blocking on-screen warning that the prediction is an extrapolation.

## Limitations

- The underlying models were trained on only 15 yearly observations (12
  training, 3 test), which is a very small sample for machine learning; all
  performance figures should be read with that in mind.
- Incidence Rate and Infection Prevalence models have negative test R² and
  should not be relied on for accurate forecasting; they are included for
  completeness and model comparison only.
- Predictions are national-level, annual, model-based estimates. They are
  not clinical diagnoses, do not account for sub-national variation, and
  should not be used for individual patient decisions or as an official
  epidemiological estimate.
- The saved pipelines were fit on the same 12-row training partition
  reported in Chapter 4 (not refit on the full 15-year dataset), so
  predictions for the 3 held-out years reproduce Chapter 4's reported test
  predictions exactly, but the models have not seen those 3 years' data.

## Project Structure

```
malaria_prediction_system/
├── app.py                  # Streamlit application (loads models, no training)
├── train_models.py         # Fits and saves the 3 winning pipelines (run once)
├── models/                 # Saved .pkl pipelines (created by train_models.py)
│   ├── incidence_xgboost.pkl
│   ├── prevalence_svr.pkl
│   └── mortality_xgboost.pkl
├── data/
│   └── analytic_dataset.csv
├── outputs/
│   └── model_performance_results.csv
├── requirements.txt
└── README.md
```
