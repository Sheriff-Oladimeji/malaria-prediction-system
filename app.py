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

Per the latest supervisor feedback, the interface is kept deliberately
minimal: enter or upload data, generate a prediction, view the result.
"""
import io
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

# Short, non-technical explanations shown in the "What do these parameters
# mean?" expander and as field tooltips. Purely descriptive text; does not
# affect what is passed to the models.
INDICATOR_HELP = {
    "Fevers U5": "Estimated number of children under 5 presenting with fever.",
    "Attendance U5 Public": "Number of children under 5 attending public health facilities.",
    "Attendance U5 Private": "Number of children under 5 attending private health facilities.",
    "Attendance O5 Private": "Number of patients aged over 5 attending private health facilities.",
    "Tested U5 Public": "Number of children under 5 tested for malaria in public health facilities.",
    "Tested U5 Private": "Number of children under 5 tested for malaria in private health facilities.",
    "Untested Received AM Public": "Number of patients receiving antimalarial medication without a recorded malaria test in public health facilities.",
    "Tested Negative Received AM Public": "Number of patients who tested negative for malaria but received antimalarial medication in public health facilities.",
    "TPR": "The proportion of malaria tests that returned a positive result (entered as a percentage).",
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

# Chapter 4 Table 4.5 results, used only for the small model/R2 caption
# shown under a prediction (not recomputed by the app).
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
    """Sensible input defaults derived from the actual 2010-2024 analytic
    dataset (data/analytic_dataset.csv), not arbitrary guesses."""
    df = pd.read_csv(DATA_PATH)
    stats = df[FEATURES].agg(["min", "mean", "max"]).T
    return stats


def count_input(feat: str, default_value: int):
    """A number_input for a count-based predictor: whole numbers only,
    minimum 0, step 1, no decimal places, no negative values."""
    return st.number_input(
        feat,
        min_value=0,
        value=int(default_value),
        step=1,
        format="%d",
        help=INDICATOR_HELP[feat],
        key=f"in_{feat}",
    )


def predict(target: str, row: pd.DataFrame) -> float:
    """Runs the existing, already-fitted pipeline for `target` on a single
    row (or many rows) of the 9 features, in FEATURES order. No training or
    fitting happens here."""
    bundle = load_model(target)
    pipeline = bundle["pipeline"]
    return bundle["model_name"], pipeline.predict(row[FEATURES])


def result_caption(target: str) -> str:
    perf = PERFORMANCE_TABLE[PERFORMANCE_TABLE["Target"] == target].iloc[0]
    note = "" if perf["Test R2"] >= 0 else " (below mean-baseline on held-out test data)"
    return f"{perf['Model']} · Test R² = {perf['Test R2']:.3f}{note}"


TEMPLATE_COLUMNS = FEATURES  # Excel column headers must match exactly.


def build_template_bytes() -> bytes:
    """One example row (2010 values, rounded) so users see valid formatting."""
    ranges = load_feature_ranges()
    example = {f: int(round(ranges.loc[f, "mean"])) for f in FEATURES if f != "TPR"}
    example["TPR"] = round(ranges.loc["TPR", "mean"], 4)
    template_df = pd.DataFrame([example])[TEMPLATE_COLUMNS]
    buffer = io.BytesIO()
    template_df.to_excel(buffer, index=False, engine="openpyxl")
    return buffer.getvalue()


def validate_excel(df: pd.DataFrame):
    """Checks required columns exist, values are numeric, and counts are
    non-negative. Returns (clean_df, error_messages, skipped_row_labels)."""
    errors = []
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        errors.append("Missing required column(s): " + ", ".join(missing))
        return None, errors, []

    work = df.copy()
    for col in FEATURES:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    bad_type_mask = work[FEATURES].isna().any(axis=1)
    negative_mask = (work[FEATURES] < 0).any(axis=1)
    invalid_mask = bad_type_mask | negative_mask

    skipped = []
    for idx in work.index[invalid_mask]:
        reason = "non-numeric value" if bad_type_mask[idx] else "negative value"
        skipped.append(f"row {idx + 2} ({reason})")  # +2: header row + 1-indexed

    clean = work.loc[~invalid_mask].reset_index(drop=True)
    return clean, errors, skipped


st.set_page_config(page_title="Pediatric Malaria Prediction System", page_icon="🩺")

st.title("Pediatric Malaria Prediction System")
st.caption("Early detection saves lives.")

target = st.selectbox("Prediction Target", options=list(TARGET_CONFIG.keys()), index=0)

models_missing = [t for t in TARGET_CONFIG if load_model(t) is None]
if models_missing:
    st.error(
        "The following trained model file(s) were not found in `models/`: "
        + ", ".join(TARGET_CONFIG[t]["file"] for t in models_missing)
        + ". Run `python3 train_models.py` once from the project folder before "
        "starting this app."
    )
    st.stop()

ranges = load_feature_ranges()

st.header("Add Data")
tab_manual, tab_excel = st.tabs(["Manual Entry", "Upload Excel"])

# ---------------------------------------------------------------------------
# Manual Parameter Entry
# ---------------------------------------------------------------------------
with tab_manual:
    st.markdown("**Required Parameters**")
    inputs = {}
    col1, col2 = st.columns(2)
    count_features = [f for f in FEATURES if f != "TPR"]
    for i, feat in enumerate(count_features):
        with (col1 if i % 2 == 0 else col2):
            inputs[feat] = count_input(feat, ranges.loc[feat, "mean"])

    auto_tpr = st.toggle(
        "Calculate TPR automatically from testing data",
        value=False,
        help="TPR = (Tested Positive Public + Tested Positive Private) / "
             "(Tested U5 Public + Tested U5 Private).",
    )
    if auto_tpr:
        c1, c2 = st.columns(2)
        tp_public = c1.number_input("Tested Positive Public", min_value=0, value=0, step=1, format="%d")
        tp_private = c2.number_input("Tested Positive Private", min_value=0, value=0, step=1, format="%d")
        denom = inputs["Tested U5 Public"] + inputs["Tested U5 Private"]
        tpr_fraction = (tp_public + tp_private) / denom if denom > 0 else 0.0
        st.metric("Computed TPR", f"{tpr_fraction * 100:.1f}%")
    else:
        tpr_pct = st.number_input(
            "TPR (%)",
            min_value=0,
            max_value=200,
            value=int(round(ranges.loc["TPR", "mean"] * 100)),
            step=1,
            format="%d",
            help=INDICATOR_HELP["TPR"],
        )
        # Convert the percentage shown to the user back to the 0-1 fraction
        # the trained pipeline was fit on (e.g. 97 -> 0.97).
        tpr_fraction = tpr_pct / 100.0
    inputs["TPR"] = tpr_fraction

    manual_clicked = st.button("Generate Prediction", type="primary", use_container_width=True, key="btn_manual")

    if manual_clicked:
        row = pd.DataFrame([inputs])[FEATURES]
        model_name, prediction = predict(target, row)
        prediction_display = int(round(float(prediction[0])))

        st.subheader("Prediction Result")
        st.markdown(f"**Predicted {target}**")
        st.markdown(
            f"<div style='font-size:2.5rem;font-weight:700;line-height:1.1'>{prediction_display:,}</div>"
            f"<div style='color:gray'>{TARGET_CONFIG[target]['unit']}</div>",
            unsafe_allow_html=True,
        )
        st.caption(result_caption(target))

# ---------------------------------------------------------------------------
# Excel Upload
# ---------------------------------------------------------------------------
with tab_excel:
    st.markdown("**Required columns** (exact names): " + ", ".join(f"`{f}`" for f in FEATURES))
    st.caption("`TPR` should be a decimal (e.g. 0.97 for 97%), matching the other numeric columns.")
    st.download_button(
        "Download template (.xlsx)",
        data=build_template_bytes(),
        file_name="malaria_prediction_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    uploaded = st.file_uploader("Upload Excel file", type=["xlsx"])

    if uploaded is not None:
        try:
            raw_df = pd.read_excel(uploaded, engine="openpyxl")
        except Exception as e:
            st.error(f"Could not read this file as an Excel (.xlsx) file: {e}")
            raw_df = None

        if raw_df is not None:
            clean_df, errors, skipped = validate_excel(raw_df)
            for msg in errors:
                st.error(msg)
            if skipped:
                st.warning("Skipped " + "; ".join(skipped) + ".")

            if clean_df is not None and len(clean_df) > 0:
                st.success(f"{len(clean_df)} valid row(s) ready for prediction.")
                excel_clicked = st.button(
                    "Generate Prediction", type="primary", use_container_width=True, key="btn_excel"
                )
                if excel_clicked:
                    model_name, predictions = predict(target, clean_df)
                    result_df = pd.DataFrame({
                        "Row": range(1, len(clean_df) + 1),
                        f"Predicted {target}": [int(round(float(p))) for p in predictions],
                    })
                    st.subheader("Prediction Result")
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                    st.caption(f"{TARGET_CONFIG[target]['unit']} · {result_caption(target)}")
            elif clean_df is not None:
                st.error("No valid rows remained after validation; nothing to predict.")

st.divider()
with st.expander("ⓘ What do these parameters mean?"):
    for feat in FEATURES:
        label = "TPR (Test Positivity Rate)" if feat == "TPR" else feat
        st.markdown(f"- **{label}**: {INDICATOR_HELP[feat]}")
