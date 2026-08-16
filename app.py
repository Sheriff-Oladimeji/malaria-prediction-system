"""
Pediatric Malaria Prediction & Decision-Support System

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

# The nine predictors the trained models expect, in this exact order.
# Unchanged from the original build: do not add, remove, or reorder these.
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

# Plain-language explanations for a non-technical public-health audience.
# Purely descriptive text; does not affect what is passed to the models.
INDICATOR_HELP = {
    "Fevers U5": "Estimated number of children under 5 presenting with fever.",
    "Attendance U5 Public": "Number of children under 5 attending public health facilities.",
    "Attendance U5 Private": "Number of children under 5 attending private health facilities.",
    "Attendance O5 Private": "Number of patients aged over 5 attending private health facilities.",
    "Tested U5 Public": "Number of children under 5 tested for malaria in public health facilities.",
    "Tested U5 Private": "Number of children under 5 tested for malaria in private health facilities.",
    "Untested Received AM Public": "Number of patients receiving antimalarial medication without a recorded malaria test in public health facilities.",
    "Tested Negative Received AM Public": "Number of patients who tested negative for malaria but received antimalarial medication in public health facilities.",
    "TPR": "The proportion of malaria tests that returned a positive result.",
}

# target -> (saved pipeline filename, display model name, units)
# Units are unchanged from the original build; sourced from the Chapter 3/4
# target definitions (Incidence Rate = cases per 1,000; Infection Prevalence
# = % of children under 5; Mortality Rate = deaths per 100,000).
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


def format_count(x: float) -> str:
    """Whole-number formatting for count-based indicators (people/encounters)."""
    return f"{x:,.0f}"


def count_input(feat: str, ranges: pd.DataFrame):
    """A number_input for a count-based predictor: whole numbers only,
    minimum 0, step 1, no decimal places, no negative values."""
    r = ranges.loc[feat]
    help_text = (
        f"{INDICATOR_HELP[feat]}\n\n"
        f"Observed 2010-2024 range: {format_count(r['min'])} to {format_count(r['max'])}."
    )
    return st.number_input(
        feat,
        min_value=0,
        value=int(round(r["mean"])),
        step=1,
        format="%d",
        help=help_text,
        key=f"in_{feat}",
    )


st.set_page_config(page_title="Pediatric Malaria Prediction System", page_icon="🩺", layout="wide")

# ---------------------------------------------------------------------------
# Title, audience, and purpose
# ---------------------------------------------------------------------------
st.title("Pediatric Malaria Prediction System")
st.subheader("Pediatric Malaria Prediction & Decision-Support System")
st.markdown(
    "Designed to support public health decision-makers, malaria programme "
    "managers, healthcare planners, and government/public health agencies in "
    "assessing potential pediatric malaria outcomes in Nigeria."
)
st.caption(
    "The system uses malaria burden and healthcare service-utilisation "
    "indicators to generate model-based estimates of pediatric malaria "
    "outcomes. It does not produce clinical diagnoses or official "
    "epidemiological estimates."
)

st.divider()
st.header("What This System Does")
st.markdown(
    "This system takes a set of national-level malaria service-utilisation "
    "indicators (fever cases, clinic attendance, testing volumes, and "
    "treatment records) and uses one of three trained machine-learning "
    "models to produce a model-based estimate of a chosen malaria outcome: "
    "**Incidence Rate**, **Infection Prevalence**, or **Mortality Rate**. "
    "The models were trained and evaluated on Nigerian national data for "
    "2010-2024, as documented in Chapter 4 of the underlying research."
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

# ---------------------------------------------------------------------------
# Understanding the Indicators
# ---------------------------------------------------------------------------
st.divider()
with st.expander("Understanding the Indicators", expanded=False):
    st.markdown(
        "Plain-language explanation of each indicator used by the models:"
    )
    for feat in FEATURES:
        label = "TPR (Test Positivity Rate)" if feat == "TPR" else feat
        st.markdown(f"**{label}**  \n{INDICATOR_HELP[feat]}")

# ---------------------------------------------------------------------------
# Select Prediction Target
# ---------------------------------------------------------------------------
st.divider()
st.header("Select Prediction Target")
target = st.selectbox(
    "Prediction Target",
    options=["Mortality Rate", "Incidence Rate", "Infection Prevalence"],
    index=0,
    help="Mortality Rate is the default because it is the only target for which "
         "any model achieved a positive test R2 in Chapter 4 (see Model "
         "Performance & Research Context below).",
)

# ---------------------------------------------------------------------------
# Enter Malaria-Related Indicators
# ---------------------------------------------------------------------------
st.divider()
st.header("Enter Malaria-Related Indicators")
st.caption(
    "Default values are the 2010-2024 dataset mean for each indicator; the min/max "
    "shown in each field's tooltip are the actual observed range in that same dataset. "
    "See 'Understanding the Indicators' above for what each field means."
)

col1, col2, col3 = st.columns(3)
inputs = {}

with col1:
    st.subheader("Fever and Healthcare Attendance")
    for feat in ["Fevers U5", "Attendance U5 Public", "Attendance U5 Private", "Attendance O5 Private"]:
        inputs[feat] = count_input(feat, ranges)

with col2:
    st.subheader("Malaria Testing")
    for feat in ["Tested U5 Public", "Tested U5 Private", "Tested Negative Received AM Public"]:
        inputs[feat] = count_input(feat, ranges)

    st.subheader("Treatment / Other Indicator")
    inputs["Untested Received AM Public"] = count_input("Untested Received AM Public", ranges)

with col3:
    st.subheader("Test Positivity Rate (TPR)")
    tpr_r = ranges.loc["TPR"]
    auto_tpr = st.toggle(
        "Calculate TPR automatically from testing data",
        value=False,
        help="TPR = (Tested Positive Public + Tested Positive Private) / "
             "(Tested U5 Public + Tested U5 Private). The denominator reuses the "
             "'Malaria Testing' values entered in the middle column.",
    )
    if auto_tpr:
        tp_public = st.number_input(
            "Tested Positive Public", min_value=0, value=0, step=1, format="%d",
            help="Number of children under 5 who tested positive for malaria in "
                 "public health facilities. Used only to compute TPR; not itself "
                 "one of the 9 model features.",
        )
        tp_private = st.number_input(
            "Tested Positive Private", min_value=0, value=0, step=1, format="%d",
            help="Number of children under 5 who tested positive for malaria in "
                 "private health facilities. Used only to compute TPR; not itself "
                 "one of the 9 model features.",
        )
        denom = inputs["Tested U5 Public"] + inputs["Tested U5 Private"]
        computed_tpr = (tp_public + tp_private) / denom if denom > 0 else 0.0
        st.metric("Computed TPR", f"{computed_tpr * 100:.1f}%")
        inputs["TPR"] = computed_tpr
    else:
        tpr_pct_default = int(round(tpr_r["mean"] * 100))
        tpr_pct = st.number_input(
            "TPR (%)",
            min_value=0,
            max_value=200,
            value=tpr_pct_default,
            step=1,
            format="%d",
            help=f"{INDICATOR_HELP['TPR']} Entered as a whole-number percentage, "
                 f"e.g. 97 means 97%. Observed 2010-2024 range: "
                 f"{tpr_r['min'] * 100:.1f}% to {tpr_r['max'] * 100:.1f}%. Values "
                 "above 100% can occur because the source counts are modelled "
                 "estimates rather than exact tallies (see Chapter 4, Section 4.2).",
            key="in_TPR_pct",
        )
        # Convert the percentage shown to the user back to the 0-1 fraction
        # the trained pipeline was fit on (e.g. 97 -> 0.97). The model always
        # receives TPR in exactly the representation used during training.
        inputs["TPR"] = tpr_pct / 100.0

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

# ---------------------------------------------------------------------------
# Generate Prediction
# ---------------------------------------------------------------------------
st.divider()
predict_clicked = st.button("Generate Prediction", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction Result
# ---------------------------------------------------------------------------
st.divider()
st.header("Prediction Result")

if predict_clicked:
    try:
        bundle = load_model(target)
        pipeline = bundle["pipeline"]
        row = pd.DataFrame([[inputs[f] for f in FEATURES]], columns=FEATURES)

        # The pipeline's raw prediction is kept unrounded; rounding happens
        # only below, at display time, and never feeds back into any
        # calculation.
        prediction = float(pipeline.predict(row)[0])
        prediction_display = int(round(prediction))

        cfg = TARGET_CONFIG[target]
        perf = PERFORMANCE_TABLE[PERFORMANCE_TABLE["Target"] == target].iloc[0]

        st.markdown(f"#### Predicted Pediatric Malaria {target}")
        st.markdown(
            f"<div style='font-size:3rem;font-weight:700;line-height:1.1'>"
            f"{prediction_display:,}</div>"
            f"<div style='font-size:1rem;color:gray'>{cfg['unit']}</div>",
            unsafe_allow_html=True,
        )

        rcol1, rcol2 = st.columns(2)
        rcol1.metric("Model Used", bundle["model_name"])
        rcol2.metric("Model Test R2", f"{perf['Test R2']:.3f}")

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

        st.markdown("**Interpretation**")
        st.markdown(
            f"Based on the indicators provided, the model estimates a pediatric "
            f"malaria {target.lower()} of approximately **{prediction_display:,}** "
            f"({cfg['unit']}). This is a model-based estimate, not a guaranteed "
            "outcome, and should be read alongside the model's reported "
            "performance above."
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.info("Enter values above and click Generate Prediction to see a result.")

# ---------------------------------------------------------------------------
# Model Performance & Research Context
# ---------------------------------------------------------------------------
st.divider()
st.header("Model Performance & Research Context")
st.caption("Held-out test-set metrics as reported in Chapter 4, Table 4.5 (not recomputed here).")
st.dataframe(PERFORMANCE_TABLE, use_container_width=True, hide_index=True)

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
  no training or retraining at prediction time. TPR is the only input transformed
  before reaching the model (percentage entered by the user divided by 100), which
  exactly reverses the percentage display and restores the 0-1 fraction the model
  was trained on.
        """
    )

# ---------------------------------------------------------------------------
# Research Prototype Disclaimer
# ---------------------------------------------------------------------------
st.divider()
st.header("Research Prototype Disclaimer")
st.markdown(
    "This is a research prototype built from a final-year academic study, not a "
    "validated clinical or public health surveillance tool. The predictions "
    "generated by this application are model-based estimates and should not be "
    "interpreted as direct clinical diagnoses or official epidemiological "
    "estimates. Model performance varies by target: the mortality model "
    "demonstrated the strongest performance on the held-out test data "
    "(R2 = 0.826), while the incidence and infection prevalence models produced "
    "negative test R2 values and are included here for model-comparison purposes "
    "rather than as reliable forecasts."
)
