"""
Pediatric Malaria Prediction System

Streamlit interface for the machine-learning models developed in Chapter 4
of "Development of a Machine Learning-Based Predictive Model for Pediatric
Malaria Prevalence in Nigeria."

This app does NOT train or retrain any model. It loads the exact fitted
scikit-learn Pipeline objects (SimpleImputer -> MinMaxScaler -> model)
produced by train_models.py, which itself reruns the identical procedure
used in Chapter 4 (same 9 features, same 80/20 split with random_state=42,
same 5-fold GridSearchCV grids) for only the three winning (target, model)
combinations already identified in Chapter 4 Table 4.6.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "analytic_dataset.csv"

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

# target -> (saved pipeline filename, display model name, units)
TARGET_CONFIG = {
    "Mortality Rate": dict(file="mortality_xgboost.pkl", unit="deaths per 100,000"),
    "Incidence Rate": dict(file="incidence_xgboost.pkl", unit="cases per 1,000"),
    "Infection Prevalence": dict(file="prevalence_svr.pkl", unit="% of children under 5"),
}

# Chapter 4 Table 4.5 results, shown as-is (these are NOT recomputed by the app).
PERFORMANCE_TABLE = pd.DataFrame([
    {"Target": "Incidence Rate", "Model": "XGBoost", "RMSE": 35.268, "MAE": 25.313, "Test R2": -0.699},
    {"Target": "Infection Prevalence", "Model": "SVR", "RMSE": 3.664, "MAE": 3.071, "Test R2": -0.344},
    {"Target": "Mortality Rate", "Model": "XGBoost", "RMSE": 7.949, "MAE": 7.013, "Test R2": 0.826},
])


@st.cache_resource
def load_model(target: str):
    # joblib.load deserializes a pickle, which can execute arbitrary code if
    # the file is untrusted. Safe here: these .pkl files are produced only by
    # train_models.py in this same project (never accepted as external/user
    # input), so the trust boundary is the developer's own machine/repo, not
    # a third party.
    path = MODELS_DIR / TARGET_CONFIG[target]["file"]
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def load_feature_ranges():
    """Sensible input ranges/defaults derived from the actual 2010-2024
    analytic dataset (data/analytic_dataset.csv), not arbitrary guesses."""
    df = pd.read_csv(DATA_PATH)
    stats = df[FEATURES].agg(["min", "mean", "max"]).T
    return stats


def format_number(x: float) -> str:
    return f"{x:,.2f}"


st.set_page_config(page_title="Pediatric Malaria Prediction System", page_icon="🩺", layout="wide")

st.title("Pediatric Malaria Prediction System")
st.caption(
    "An interactive machine-learning system for predicting malaria-related outcomes "
    "in Nigeria using malaria burden and healthcare service-utilisation indicators."
)

ranges = load_feature_ranges()

models_missing = [t for t in TARGET_CONFIG if load_model(t) is None]
if models_missing:
    st.error(
        "The following trained model file(s) were not found in `models/`: "
        + ", ".join(TARGET_CONFIG[t]["file"] for t in models_missing)
        + ". Run `python3 train_models.py` once from the project folder before "
        "starting this app."
    )
    st.stop()

st.divider()
st.header("Enter Malaria-Related Indicators")
st.caption(
    "Default values are the 2010-2024 dataset mean for each indicator; the min/max "
    "shown next to each field are the actual observed range in that same dataset."
)

col1, col2, col3 = st.columns(3)
inputs = {}

with col1:
    st.subheader("Fever and Healthcare Attendance")
    for feat in ["Fevers U5", "Attendance U5 Public", "Attendance U5 Private", "Attendance O5 Private"]:
        r = ranges.loc[feat]
        inputs[feat] = st.number_input(
            feat,
            min_value=0.0,
            value=float(round(r["mean"], 2)),
            step=float(round(r["mean"] * 0.01, 2)) or 1.0,
            help=f"Observed 2010-2024 range: {format_number(r['min'])} to {format_number(r['max'])}",
            key=f"in_{feat}",
        )

with col2:
    st.subheader("Malaria Testing")
    for feat in ["Tested U5 Public", "Tested U5 Private", "Tested Negative Received AM Public"]:
        r = ranges.loc[feat]
        inputs[feat] = st.number_input(
            feat,
            min_value=0.0,
            value=float(round(r["mean"], 2)),
            step=float(round(r["mean"] * 0.01, 2)) or 1.0,
            help=f"Observed 2010-2024 range: {format_number(r['min'])} to {format_number(r['max'])}",
            key=f"in_{feat}",
        )

    st.subheader("Treatment / Other Indicator")
    feat = "Untested Received AM Public"
    r = ranges.loc[feat]
    inputs[feat] = st.number_input(
        feat,
        min_value=0.0,
        value=float(round(r["mean"], 2)),
        step=float(round(r["mean"] * 0.01, 2)) or 1.0,
        help=f"Observed 2010-2024 range: {format_number(r['min'])} to {format_number(r['max'])}",
        key=f"in_{feat}",
    )

with col3:
    st.subheader("Test Positivity Rate (TPR)")
    tpr_r = ranges.loc["TPR"]
    auto_tpr = st.toggle(
        "Calculate TPR automatically from testing data",
        value=False,
        help="TPR = (Tested Positive Public + Tested Positive Private) / "
             "(Tested U5 Public + Tested U5 Private). The denominator reuses the "
             "'Malaria Testing' values entered on the left.",
    )
    if auto_tpr:
        tp_public = st.number_input(
            "Tested Positive Public", min_value=0.0, value=0.0, step=1000.0,
            help="Used only to compute TPR; not itself one of the 9 model features.",
        )
        tp_private = st.number_input(
            "Tested Positive Private", min_value=0.0, value=0.0, step=1000.0,
            help="Used only to compute TPR; not itself one of the 9 model features.",
        )
        denom = inputs["Tested U5 Public"] + inputs["Tested U5 Private"]
        computed_tpr = (tp_public + tp_private) / denom if denom > 0 else 0.0
        st.metric("Computed TPR", f"{computed_tpr:.4f}")
        inputs["TPR"] = computed_tpr
    else:
        inputs["TPR"] = st.number_input(
            "TPR",
            min_value=0.0,
            max_value=2.0,
            value=float(round(tpr_r["mean"], 4)),
            step=0.01,
            help=f"Observed 2010-2024 range: {tpr_r['min']:.4f} to {tpr_r['max']:.4f}. "
                 "Values above 1.0 can occur because the source counts are modelled "
                 "estimates rather than exact tallies (see Chapter 4, Section 4.2).",
            key="in_TPR",
        )

# Warn (do not block) when an input falls outside the observed training range,
# since the model is extrapolating in that case.
out_of_range = [
    feat for feat in FEATURES
    if not (ranges.loc[feat, "min"] * 0.5 <= inputs[feat] <= ranges.loc[feat, "max"] * 1.5)
]
if out_of_range:
    st.warning(
        "The following inputs are far outside the 2010-2024 observed range the "
        "models were trained on, so predictions for them are an extrapolation and "
        "less trustworthy: " + ", ".join(out_of_range)
    )

st.divider()
st.header("Select Prediction Target")
target = st.selectbox(
    "Prediction Target",
    options=["Mortality Rate", "Incidence Rate", "Infection Prevalence"],
    index=0,
    help="Mortality Rate is the default because it is the only target for which "
         "any model achieved a positive test R2 in Chapter 4 (see Model Performance below).",
)

predict_clicked = st.button("Predict", type="primary", use_container_width=True)

st.divider()
st.header("Prediction Result")

if predict_clicked:
    try:
        bundle = load_model(target)
        pipeline = bundle["pipeline"]
        row = pd.DataFrame([[inputs[f] for f in FEATURES]], columns=FEATURES)
        prediction = float(pipeline.predict(row)[0])

        cfg = TARGET_CONFIG[target]
        perf = PERFORMANCE_TABLE[PERFORMANCE_TABLE["Target"] == target].iloc[0]

        rcol1, rcol2, rcol3 = st.columns(3)
        rcol1.metric(f"Predicted {target}", f"{prediction:,.2f}", help=cfg["unit"])
        rcol2.metric("Model Used", bundle["model_name"])
        rcol3.metric("Model Test R2", f"{perf['Test R2']:.3f}")

        if perf["Test R2"] < 0:
            st.warning(
                f"Model Test R2 for {target} is negative ({perf['Test R2']:.3f}), meaning "
                "this model did not outperform a naive baseline that always predicts the "
                "training-set mean on the held-out test years (2010, 2019, 2021). Treat this "
                "prediction as illustrative of the deployed pipeline, not as an accurate "
                "forecast. See Chapter 4, Section 4.6-4.8 for the full discussion."
            )
        else:
            st.success(
                f"This is the strongest-performing model in the study (Test R2 = "
                f"{perf['Test R2']:.3f}, RMSE = {perf['RMSE']:.3f}, MAE = {perf['MAE']:.3f})."
            )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.info("Enter values above and click Predict to generate a result.")

st.divider()
st.header("Model Performance")
st.caption("Held-out test-set metrics as reported in Chapter 4, Table 4.5 (not recomputed here).")
st.dataframe(PERFORMANCE_TABLE, use_container_width=True, hide_index=True)

st.divider()
st.header("Model Interpretation")
st.markdown(
    "The predictions generated by this application are model-based estimates and "
    "should not be interpreted as direct clinical diagnoses or official epidemiological "
    "estimates. Model performance varies by target: the mortality model demonstrated "
    "the strongest performance on the held-out test data (R2 = 0.826), while the "
    "incidence and infection prevalence models produced negative test R2 values and "
    "are included here for model-comparison purposes rather than as reliable forecasts."
)

with st.expander("About the underlying models and pipeline"):
    st.markdown(
        f"""
- **Features used** ({len(FEATURES)}, in this exact order): {', '.join(FEATURES)}
- **Preprocessing**: `SimpleImputer(strategy="mean")` then `MinMaxScaler()`, fit only on the
  training partition during Chapter 4's model development, saved as part of the same
  scikit-learn `Pipeline` as the model itself.
- **Training/validation**: 80/20 train-test split (`random_state=42`), 5-fold
  `GridSearchCV` hyperparameter tuning on the training partition only.
- **Dataset**: 15 yearly observations (2010-2024), 12 used for training, 3 held out
  for testing (2010, 2019, 2021).
- This application loads the already-fitted pipelines from `models/*.pkl` and performs
  no training or retraining at prediction time.
        """
    )
