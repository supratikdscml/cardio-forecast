import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cardio Forecast", page_icon="🫀", layout="wide")

MODEL_PATH = Path("models/forecaster_model.joblib")
METADATA_PATH = Path("models/model_metadata.json")
RAW_COLUMNS = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
]
RISK_BANDS = [
    (0.00, 0.25, "Low"),
    (0.25, 0.50, "Moderate"),
    (0.50, 0.75, "High"),
    (0.75, 1.01, "Very High"),
]
DEFAULT_PATIENT = {
    "Age": 58,
    "Sex": "M",
    "ChestPainType": "ASY",
    "RestingBP": 145,
    "Cholesterol": 260,
    "FastingBS": 1,
    "RestingECG": "ST",
    "MaxHR": 122,
    "ExerciseAngina": "Y",
    "Oldpeak": 2.3,
    "ST_Slope": "Flat",
}

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #07111f 0%, #0f172a 100%); }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    div[data-testid="stSidebar"] { background: rgba(15, 23, 42, 0.95); }
    .stMetric [data-testid="stMetricValue"] { font-size: 1.8rem; }
    .risk-badge { font-size: 1.15rem; font-weight: 600; padding: 0.35rem 0.65rem; border-radius: 999px; display: inline-block; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_payload():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model artifacts not found. Run export_models.py first.")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        return None
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _risk_band(score: float) -> str:
    for lo, hi, label in RISK_BANDS:
        if lo <= score < hi:
            return label
    return RISK_BANDS[-1][2]


def _risk_color(band: str) -> str:
    palette = {"Low": "#22c55e", "Moderate": "#f59e0b", "High": "#f97316", "Very High": "#ef4444"}
    return palette.get(band, "#60a5fa")


def to_patient_frame(patient: dict) -> pd.DataFrame:
    missing = [c for c in RAW_COLUMNS if c not in patient]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    row = {k: patient[k] for k in RAW_COLUMNS}
    df = pd.DataFrame([row])
    if not pd.api.types.is_numeric_dtype(df["Sex"]):
        df["Sex"] = df["Sex"].map({"M": 1, "F": 0})
    if not pd.api.types.is_numeric_dtype(df["ExerciseAngina"]):
        df["ExerciseAngina"] = df["ExerciseAngina"].map({"Y": 1, "N": 0})
    df["RestingBP"] = df["RestingBP"].replace(0, np.nan)
    df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)
    return df


def local_explanation(patient_df: pd.DataFrame, logreg_model, feature_names: list, unified_importance: pd.Series, top_n: int = 5):
    prep = logreg_model.named_steps["prep"]
    clf = logreg_model.named_steps["clf"]
    X_transformed = prep.transform(patient_df)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    contrib = X_transformed[0] * clf.coef_[0]
    contrib_df = pd.DataFrame({"feature": feature_names, "local_effect": contrib})
    contrib_df = contrib_df.set_index("feature")
    contrib_df["global_importance"] = unified_importance.reindex(contrib_df.index).fillna(0)
    contrib_df["driver_score"] = contrib_df["global_importance"] * contrib_df["local_effect"].abs()
    contrib_df["direction"] = np.where(contrib_df["local_effect"] > 0, "increases risk", "lowers risk")
    top = contrib_df.sort_values("driver_score", ascending=False).head(top_n)
    return top[["direction", "local_effect", "global_importance"]]


def predict_risk(patient: dict):
    payload = load_payload()
    models = payload["models"]
    weights = payload["weights"]
    unified_importance = payload["unified_importance"]
    feature_names = payload["feature_names"]
    logreg_model = models["Logistic Regression"]

    patient_df = to_patient_frame(patient)
    per_model = {}
    weighted_sum = 0.0
    for name, mdl in models.items():
        proba = float(mdl.predict_proba(patient_df)[:, 1][0])
        per_model[name] = proba
        weighted_sum += proba * weights[name]

    risk_score = float(weighted_sum)
    risk_percent = risk_score * 100
    band = _risk_band(risk_score)
    drivers = local_explanation(patient_df, logreg_model, feature_names, unified_importance)

    return {
        "risk_score": risk_score,
        "risk_percentage": risk_percent,
        "risk_band": band,
        "per_model_probability": per_model,
        "model_weights_used": {k: float(v) for k, v in weights.items()},
        "top_personal_drivers": drivers.reset_index(),
    }


def render_header():
    st.title("Cardio Forecast")
    st.caption("Enterprise-grade cardiovascular risk assessment for clinical decision support.")
    st.markdown(
        """
        This application combines an ensemble of trained models with explainable AI to provide a
        transparent and auditable heart disease risk profile for a patient.
        """
    )


def render_sidebar(patient_state):
    with st.sidebar:
        st.header("Clinical intake")
        st.caption("Capture the patient profile for the forecast")

        if st.button("Load demo profile", use_container_width=True):
            st.session_state["patient_profile"] = DEFAULT_PATIENT.copy()

        with st.form("risk_form"):
            age = st.slider("Age (years)", 28, 77, int(patient_state.get("Age", DEFAULT_PATIENT["Age"])))
            sex = st.radio("Sex", ["M", "F"], index=0 if patient_state.get("Sex", DEFAULT_PATIENT["Sex"]) == "M" else 1, horizontal=True)
            chest_pain = st.selectbox(
                "Chest pain type",
                ["TA", "ATA", "NAP", "ASY"],
                index=["TA", "ATA", "NAP", "ASY"].index(patient_state.get("ChestPainType", DEFAULT_PATIENT["ChestPainType"])),
            )
            resting_bp = st.slider("Resting BP", 80, 200, int(patient_state.get("RestingBP", DEFAULT_PATIENT["RestingBP"])))
            cholesterol = st.slider("Cholesterol", 85, 603, int(patient_state.get("Cholesterol", DEFAULT_PATIENT["Cholesterol"])))
            fasting_bs = st.radio(
                "Fasting blood sugar > 120 mg/dl",
                [0, 1],
                index=int(patient_state.get("FastingBS", DEFAULT_PATIENT["FastingBS"])),
                horizontal=True,
            )
            resting_ecg = st.selectbox(
                "Resting ECG",
                ["Normal", "ST", "LVH"],
                index=["Normal", "ST", "LVH"].index(patient_state.get("RestingECG", DEFAULT_PATIENT["RestingECG"])),
            )
            max_hr = st.slider("Max HR", 60, 202, int(patient_state.get("MaxHR", DEFAULT_PATIENT["MaxHR"])))
            exercise_angina = st.radio(
                "Exercise-induced angina",
                ["Y", "N"],
                index=0 if patient_state.get("ExerciseAngina", DEFAULT_PATIENT["ExerciseAngina"]) == "Y" else 1,
                horizontal=True,
            )
            oldpeak = st.slider("Oldpeak", -2.6, 6.2, float(patient_state.get("Oldpeak", DEFAULT_PATIENT["Oldpeak"])), step=0.1)
            st_slope = st.selectbox(
                "ST Slope",
                ["Up", "Flat", "Down"],
                index=["Up", "Flat", "Down"].index(patient_state.get("ST_Slope", DEFAULT_PATIENT["ST_Slope"])),
            )
            submitted = st.form_submit_button("Run forecast", use_container_width=True)

        return {
            "Age": int(age),
            "Sex": sex,
            "ChestPainType": chest_pain,
            "RestingBP": int(resting_bp),
            "Cholesterol": int(cholesterol),
            "FastingBS": int(fasting_bs),
            "RestingECG": resting_ecg,
            "MaxHR": int(max_hr),
            "ExerciseAngina": exercise_angina,
            "Oldpeak": float(oldpeak),
            "ST_Slope": st_slope,
        }, submitted


def render_forecast_tab(patient_data):
    with st.spinner("Running the ensemble forecast..."):
        result = predict_risk(patient_data)

    st.success("Forecast generated successfully.")

    col_a, col_b = st.columns([1.2, 0.8])
    with col_a:
        band = result["risk_band"]
        risk_pct = result["risk_percentage"]
        color = _risk_color(band)
        st.metric("Ensemble risk score", f"{risk_pct:.1f}%")
        st.markdown(f'<div class="risk-badge" style="color:{color}; border: 1px solid {color};">Risk band: {band}</div>', unsafe_allow_html=True)
        st.progress(min(risk_pct / 100, 1.0))

        with st.expander("Clinical interpretation", expanded=True):
            st.write(
                "A higher score suggests stronger evidence for cardiovascular disease risk based on the weighted ensemble."
            )

    with col_b:
        st.subheader("Model probability view")
        prob_df = pd.DataFrame(
            {"Model": list(result["per_model_probability"].keys()), "Probability": [value * 100 for value in result["per_model_probability"].values()]}
        )
        st.bar_chart(prob_df.set_index("Model"), use_container_width=True)

    st.subheader("Top personal drivers")
    drivers_df = result["top_personal_drivers"].copy()
    drivers_df["local_effect"] = drivers_df["local_effect"].round(3)
    drivers_df["global_importance"] = drivers_df["global_importance"].round(3)
    drivers_df = drivers_df.rename(columns={"feature": "Driver", "direction": "Direction", "local_effect": "Local effect", "global_importance": "Global importance"})
    st.dataframe(drivers_df, use_container_width=True, hide_index=True)


def render_analytics_tab():
    st.subheader("Model analytics")
    st.caption("Statistical comparison, model weights, and feature importance from the exported model metadata.")

    metadata = load_metadata()
    if not metadata:
        st.warning("No model metadata is available yet. Run export_models.py to generate the evaluation artifacts.")
        return

    col1, col2, col3 = st.columns(3)
    p_value = metadata.get("friedman_p")
    with col1:
        st.metric("Friedman p-value", f"{p_value:.4f}" if p_value is not None else "n/a")
    with col2:
        st.metric("Model count", len(metadata.get("model_weights", {})))
    with col3:
        st.metric("Feature importance entries", len(metadata.get("unified_importance", {})))

    weights = metadata.get("model_weights", {})
    if weights:
        weight_df = pd.DataFrame({"Model": list(weights.keys()), "Weight": list(weights.values())}).sort_values("Weight", ascending=False)
        st.subheader("Ensemble weights")
        st.bar_chart(weight_df.set_index("Model"), use_container_width=True)

    importance = metadata.get("unified_importance", {})
    if importance:
        importance_df = pd.DataFrame({"Feature": list(importance.keys()), "Importance": list(importance.values())}).sort_values("Importance", ascending=False)
        st.subheader("Unified feature importance")
        st.bar_chart(importance_df.set_index("Feature"), use_container_width=True)


def main():
    if not MODEL_PATH.exists():
        st.error("Model artifacts were not found. Run export_models.py first to train and export the forecasting pipeline.")
        st.stop()

    if "patient_profile" not in st.session_state:
        st.session_state["patient_profile"] = DEFAULT_PATIENT.copy()

    render_header()
    patient_state = st.session_state["patient_profile"]
    patient_data, submitted = render_sidebar(patient_state)

    if submitted:
        st.session_state["patient_profile"] = patient_data
        render_forecast_tab(patient_data)
    else:
        st.info("Use the sidebar to enter or adjust a patient profile, then run the forecast.")

    st.divider()
    render_analytics_tab()


if __name__ == "__main__":
    main()
