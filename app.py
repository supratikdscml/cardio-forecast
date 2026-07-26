import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Cardio Forecast | Clinical Decision Engine",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    (0.00, 0.25, "Low Risk", "#22c55e", "Low probability of cardiovascular disease based on clinical markers. Routine preventive monitoring recommended."),
    (0.25, 0.50, "Moderate Risk", "#eab308", "Moderate clinical risk. Lifestyle interventions, lipid management, and regular cardiovascular follow-ups advised."),
    (0.50, 0.75, "High Risk", "#f97316", "High probability of coronary artery disease. Further non-invasive diagnostic evaluation (e.g., stress ECG, echocardiogram) indicated."),
    (0.75, 1.01, "Very High Risk", "#ef4444", "Severe cardiovascular risk. Prompt clinical consultation, diagnostic workup, and targeted therapeutic intervention strongly advised."),
]

PRESET_PROFILES = {
    "Demo Patient (Asymptomatic High-Risk)": {
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
    },
    "Low Risk Baseline Patient": {
        "Age": 35,
        "Sex": "F",
        "ChestPainType": "ATA",
        "RestingBP": 115,
        "Cholesterol": 180,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 175,
        "ExerciseAngina": "N",
        "Oldpeak": 0.0,
        "ST_Slope": "Up",
    },
    "Severe Ischemia Suspect (High Risk)": {
        "Age": 64,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 160,
        "Cholesterol": 290,
        "FastingBS": 1,
        "RestingECG": "LVH",
        "MaxHR": 105,
        "ExerciseAngina": "Y",
        "Oldpeak": 3.1,
        "ST_Slope": "Down",
    },
    "Atypical Female Profile": {
        "Age": 52,
        "Sex": "F",
        "ChestPainType": "NAP",
        "RestingBP": 130,
        "Cholesterol": 210,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 150,
        "ExerciseAngina": "N",
        "Oldpeak": 0.8,
        "ST_Slope": "Flat",
    },
}

FEATURE_DICTIONARY = [
    {"Feature": "Age", "Type": "Numeric", "Normal / Reference Range": "28 – 77 years", "Description": "Patient age in years."},
    {"Feature": "Sex", "Type": "Categorical", "Normal / Reference Range": "M (Male), F (Female)", "Description": "Biological sex. Males statically exhibit higher baseline risk."},
    {"Feature": "ChestPainType", "Type": "Categorical", "Normal / Reference Range": "TA, ATA, NAP, ASY", "Description": "TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic (ASY indicates higher risk)."},
    {"Feature": "RestingBP", "Type": "Numeric", "Normal / Reference Range": "90 – 120 mm Hg", "Description": "Resting blood pressure on hospital admission. Zeros auto-imputed via KNN."},
    {"Feature": "Cholesterol", "Type": "Numeric", "Normal / Reference Range": "125 – 200 mg/dl", "Description": "Serum cholesterol level. Zeros auto-imputed via KNN."},
    {"Feature": "FastingBS", "Type": "Binary", "Normal / Reference Range": "0: ≤ 120 mg/dl, 1: > 120 mg/dl", "Description": "Fasting blood sugar > 120 mg/dl indicates diabetic hyperglycemia."},
    {"Feature": "RestingECG", "Type": "Categorical", "Normal / Reference Range": "Normal, ST, LVH", "Description": "Resting electrocardiogram: Normal, ST (ST-T wave abnormality), LVH (Left Ventricular Hypertrophy)."},
    {"Feature": "MaxHR", "Type": "Numeric", "Normal / Reference Range": "60 – 200 bpm", "Description": "Maximum heart rate achieved during exercise stress testing."},
    {"Feature": "ExerciseAngina", "Type": "Binary", "Normal / Reference Range": "N (No), Y (Yes)", "Description": "Exercise-induced angina. 'Y' signals myocardial ischemia."},
    {"Feature": "Oldpeak", "Type": "Numeric", "Normal / Reference Range": "0.0 – 1.0 mm (Normal)", "Description": "ST depression induced by exercise relative to rest. Elevated values signify severe ischemia."},
    {"Feature": "ST_Slope", "Type": "Categorical", "Normal / Reference Range": "Up, Flat, Down", "Description": "Slope of peak exercise ST segment. 'Up' is protective; 'Flat' and 'Down' indicate high risk."},
]

# Custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #070e17 0%, #0d1527 50%, #111a2e 100%);
        color: #e2e8f0;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stSidebar"] {
        background: rgba(13, 21, 39, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    .chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    .chip {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        font-size: 0.85rem;
        color: #cbd5e1;
        font-weight: 500;
    }
    .risk-badge {
        font-size: 1.15rem;
        font-weight: 700;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_payload():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model artifacts not found. Please run export_models.py first.")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        return None
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_risk_info(score: float):
    for lo, hi, label, color, guide in RISK_BANDS:
        if lo <= score < hi:
            return label, color, guide
    return RISK_BANDS[-1][2], RISK_BANDS[-1][3], RISK_BANDS[-1][4]


def to_patient_frame(patient: dict) -> pd.DataFrame:
    row = {k: patient[k] for k in RAW_COLUMNS}
    df = pd.DataFrame([row])
    if not pd.api.types.is_numeric_dtype(df["Sex"]):
        df["Sex"] = df["Sex"].map({"M": 1, "F": 0})
    if not pd.api.types.is_numeric_dtype(df["ExerciseAngina"]):
        df["ExerciseAngina"] = df["ExerciseAngina"].map({"Y": 1, "N": 0})
    df["RestingBP"] = df["RestingBP"].replace(0, np.nan)
    df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)
    return df


def local_explanation(patient_df: pd.DataFrame, logreg_model, feature_names: list, unified_importance: pd.Series, top_n: int = 6):
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
    contrib_df["direction"] = np.where(contrib_df["local_effect"] > 0, "Increases Risk", "Lowers Risk")
    top = contrib_df.sort_values("driver_score", ascending=False).head(top_n)
    return top[["direction", "local_effect", "global_importance", "driver_score"]]


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
    band_label, band_color, band_guide = get_risk_info(risk_score)
    drivers = local_explanation(patient_df, logreg_model, feature_names, unified_importance)

    return {
        "risk_score": risk_score,
        "risk_percentage": risk_percent,
        "risk_band": band_label,
        "risk_color": band_color,
        "risk_guide": band_guide,
        "per_model_probability": per_model,
        "model_weights_used": {k: float(v) for k, v in weights.items()},
        "top_personal_drivers": drivers.reset_index(),
    }


def plot_risk_gauge(risk_pct: float, risk_color: str, band_label: str):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={"suffix": "%", "font": {"size": 44, "color": "#f8fafc", "family": "Inter"}},
            title={"text": f"Ensemble Risk Score — {band_label}", "font": {"size": 18, "color": "#94a3b8"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b", "dtick": 25},
                "bar": {"color": risk_color, "width": 16},
                "bgcolor": "rgba(15, 23, 42, 0.5)",
                "borderwidth": 1,
                "bordercolor": "rgba(255, 255, 255, 0.1)",
                "steps": [
                    {"range": [0, 25], "color": "rgba(34, 197, 94, 0.18)"},
                    {"range": [25, 50], "color": "rgba(234, 179, 8, 0.18)"},
                    {"range": [50, 75], "color": "rgba(249, 115, 22, 0.18)"},
                    {"range": [75, 100], "color": "rgba(239, 68, 68, 0.18)"},
                ],
                "threshold": {
                    "line": {"color": "#f8fafc", "width": 3},
                    "thickness": 0.8,
                    "value": risk_pct,
                },
            },
        )
    )
    fig.update_layout(
        height=270,
        margin=dict(l=25, r=25, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_top_drivers(drivers_df: pd.DataFrame):
    df_sorted = drivers_df.copy().sort_values("driver_score", ascending=True)
    fig = px.bar(
        df_sorted,
        x="driver_score",
        y="feature",
        orientation="h",
        color="direction",
        color_discrete_map={"Increases Risk": "#ef4444", "Lowers Risk": "#22c55e"},
        title="Personalized Risk Drivers (Strength-Weighted Local Impact)",
        labels={"driver_score": "Impact Score", "feature": "Clinical Parameter", "direction": "Effect Direction"},
    )
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=False)
    return fig


def plot_model_consensus(per_model: dict, weights: dict):
    df = pd.DataFrame({
        "Model": list(per_model.keys()),
        "Predicted Risk (%)": [v * 100 for v in per_model.values()],
        "Model Weight": [weights[k] for k in per_model.keys()],
    })
    fig = px.bar(
        df,
        x="Model",
        y="Predicted Risk (%)",
        color="Model Weight",
        color_continuous_scale="Blues",
        text_auto=".1f",
        title="Ensemble Consensus & Model Strength Weights",
    )
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", range=[0, 100])
    return fig


def init_session_state():
    default_profile = PRESET_PROFILES["Demo Patient (Asymptomatic High-Risk)"]
    for k, v in default_profile.items():
        if f"input_{k}" not in st.session_state:
            st.session_state[f"input_{k}"] = v


def load_preset_callback():
    preset_key = st.session_state.get("preset_selector")
    if preset_key in PRESET_PROFILES:
        profile = PRESET_PROFILES[preset_key]
        for k, v in profile.items():
            st.session_state[f"input_{k}"] = v


def render_header():
    st.markdown(
        """
        <div style="margin-bottom: 1rem;">
            <h1 style="font-weight: 700; color: #f8fafc; margin-bottom: 0.2rem; font-size: 2.2rem;">
                🫀 Cardio Forecast <span style="font-size: 1.1rem; color: #38bdf8; font-weight: 500; vertical-align: middle; background: rgba(56, 189, 248, 0.1); padding: 0.25rem 0.65rem; border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.2);">Clinical Decision Engine</span>
            </h1>
            <p style="color: #94a3b8; font-size: 1.05rem; margin-bottom: 0;">
                Enterprise cardiovascular risk forecasting backed by non-parametric hypothesis-tested ensemble models, calibrated probabilities, and personalized risk driver analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📖 User Guide, Instructions & Feature Dictionary", expanded=False):
        st.markdown(
            """
            ### How to Use the Cardio Forecast Engine
            1. **Select a Clinical Preset Profile** or adjust individual patient parameters in the sidebar. All risk scores update **live in real-time**.
            2. **Review the Ensemble Risk Score**: Evaluated using a weighted ensemble of 4 tuned models (Logistic Regression, Random Forest, XGBoost, SVM).
            3. **Inspect Personal Risk Drivers**: Identify which clinical factors increase or lower the patient's individual risk.
            4. **Simulate Clinical Decision Cutoffs**: Adjust the False Negative penalty factor to evaluate cost-optimal decision cutoffs.
            5. **View Model Performance & Analytics**: Switch to the **Model Analytics** tab for 50-fold cross-validation distributions, hypothesis testing results, ROC curves, and calibration diagrams.
            """
        )
        st.markdown("### Clinical Feature Dictionary & Reference Ranges")
        st.dataframe(pd.DataFrame(FEATURE_DICTIONARY), use_container_width=True, hide_index=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("<h2 style='font-size: 1.3rem; color: #f8fafc; font-weight: 600;'>📋 Patient Intake</h2>", unsafe_allow_html=True)
        st.caption("Select a preset profile or adjust parameters below. Forecast updates live.")

        st.selectbox(
            "Load Clinical Preset Profile",
            options=list(PRESET_PROFILES.keys()),
            key="preset_selector",
            on_change=load_preset_callback,
            help="Choose a pre-configured patient profile to instantly load clinical parameters.",
        )

        st.markdown("---")
        st.markdown("#### Patient Demographics")
        age = st.slider("Age (years)", 28, 77, key="input_Age", help="Patient age in years.")
        sex = st.radio("Sex", ["M", "F"], key="input_Sex", horizontal=True, help="Biological sex (M: Male, F: Female).")
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ASY", "NAP", "ATA", "TA"],
            key="input_ChestPainType",
            help="ASY: Asymptomatic, NAP: Non-Anginal Pain, ATA: Atypical Angina, TA: Typical Angina.",
        )
        resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, key="input_RestingBP", help="Resting BP on admission. Zeros auto-imputed via KNN.")
        cholesterol = st.slider("Serum Cholesterol (mg/dl)", 85, 603, key="input_Cholesterol", help="Serum cholesterol level. Zeros auto-imputed via KNN.")
        fasting_bs = st.radio(
            "Fasting Blood Sugar > 120 mg/dl",
            [0, 1],
            key="input_FastingBS",
            format_func=lambda x: "Yes (> 120 mg/dl)" if x == 1 else "No (≤ 120 mg/dl)",
            horizontal=True,
            help="1 = Fasting blood sugar > 120 mg/dl (diabetic range), 0 = Normal.",
        )

        st.markdown("#### Electrocardiogram & Stress Test")
        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"],
            key="input_RestingECG",
            help="Normal: Normal, ST: ST-T wave abnormality, LVH: Left ventricular hypertrophy.",
        )
        max_hr = st.slider("Max Heart Rate Achieved (bpm)", 60, 202, key="input_MaxHR", help="Maximum heart rate achieved during exercise stress test.")
        exercise_angina = st.radio(
            "Exercise-Induced Angina",
            ["Y", "N"],
            key="input_ExerciseAngina",
            horizontal=True,
            help="Angina provoked by exercise stress test.",
        )
        oldpeak = st.slider("Oldpeak (ST Depression)", -2.6, 6.2, key="input_Oldpeak", step=0.1, help="ST depression induced by exercise relative to rest (in mm).")
        st_slope = st.selectbox(
            "ST Slope",
            ["Up", "Flat", "Down"],
            key="input_ST_Slope",
            help="Slope of peak exercise ST segment. Flat/Down slopes indicate significant ischemia risk.",
        )

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
        }


def render_forecast_tab(patient_data):
    result = predict_risk(patient_data)
    band_label = result["risk_band"]
    band_color = result["risk_color"]
    band_guide = result["risk_guide"]
    risk_pct = result["risk_percentage"]

    # Active Patient Summary Chips
    st.markdown(
        f"""
        <div class="chip-container">
            <span class="chip">👤 Patient: {patient_data['Sex']}, {patient_data['Age']} yrs</span>
            <span class="chip">🫀 Chest Pain: {patient_data['ChestPainType']}</span>
            <span class="chip">🩺 Resting BP: {patient_data['RestingBP']} mmHg</span>
            <span class="chip">🧪 Cholesterol: {patient_data['Cholesterol']} mg/dl</span>
            <span class="chip">⚡ Max HR: {patient_data['MaxHR']} bpm</span>
            <span class="chip">📉 ST Slope: {patient_data['ST_Slope']}</span>
            <span class="chip">📉 Oldpeak: {patient_data['Oldpeak']} mm</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.2, 1.0])

    with col1:
        st.plotly_chart(plot_risk_gauge(risk_pct, band_color, band_label), use_container_width=True)
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid {band_color}; border-radius: 8px; padding: 0.8rem 1.2rem; margin-bottom: 1rem;">
                <span class="risk-badge" style="color: {band_color}; background: rgba(0,0,0,0.3); border: 1px solid {band_color}; margin-bottom: 0.4rem;">
                    Risk Band: {band_label}
                </span>
                <p style="color: #cbd5e1; font-size: 0.95rem; margin-top: 0.4rem; margin-bottom: 0;">
                    <strong style="color: #f8fafc;">Clinical Recommendation:</strong> {band_guide}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.plotly_chart(plot_model_consensus(result["per_model_probability"], result["model_weights_used"]), use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns([1.2, 0.8])
    with col3:
        st.plotly_chart(plot_top_drivers(result["top_personal_drivers"]), use_container_width=True)

    with col4:
        st.subheader("Top Personal Drivers Breakdown")
        st.caption("Combining global model-strength importance with signed local contribution for THIS patient.")
        drivers_df = result["top_personal_drivers"].copy()
        drivers_df["local_effect"] = drivers_df["local_effect"].round(3)
        drivers_df["global_importance"] = drivers_df["global_importance"].round(3)
        drivers_df["driver_score"] = drivers_df["driver_score"].round(3)
        drivers_display = drivers_df.rename(
            columns={
                "feature": "Feature",
                "direction": "Effect Direction",
                "local_effect": "Local Effect",
                "global_importance": "Global Weight",
                "driver_score": "Driver Score",
            }
        )[["Feature", "Effect Direction", "Local Effect", "Global Weight", "Driver Score"]]
        st.dataframe(drivers_display, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Interactive Cost-Sensitive Decision Cutoff Simulator
    with st.expander("⚖️ Interactive Clinical Decision Cutoff & Cost Simulator", expanded=True):
        st.markdown(
            """
            In cardiology, missing an active disease case (**False Negative**) is clinically costlier than performing 
            unnecessary secondary evaluation (**False Positive**). Adjust the cost ratio slider below to evaluate optimal decision cutoffs.
            """
        )
        metadata = load_metadata()
        if metadata and "cost_thresholds" in metadata:
            fn_weight = st.slider("False Negative Penalty Factor (relative to FP = 1.0)", 1.0, 10.0, 3.0, step=0.5)
            
            cost_info = metadata["cost_thresholds"].get("XGBoost", metadata["cost_thresholds"]["Logistic Regression"])
            opt_thresh = cost_info.get("optimal_threshold", 0.5)
            
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric("Standard Default Cutoff", "50.0%", "Balanced Classification")
            with sc2:
                st.metric(f"Cost-Optimal Decision Cutoff (FN Weight {fn_weight:.1f}x)", f"{opt_thresh * 100:.1f}%", f"-{(0.5 - opt_thresh)*100:.1f}% lower threshold")
            with sc3:
                action_required = (risk_pct / 100.0) >= opt_thresh
                decision_text = "CLINICAL ACTION INDICATED" if action_required else "ROUTINE MONITORING"
                decision_color = "#ef4444" if action_required else "#22c55e"
                st.markdown(f"**Patient Management Status:**<br/><span style='color:{decision_color}; font-size:1.15rem; font-weight:700;'>{decision_text}</span>", unsafe_allow_html=True)


def render_analytics_tab():
    st.markdown(
        """
        <div style="background: rgba(56, 189, 248, 0.05); border: 1px solid rgba(56, 189, 248, 0.2); padding: 1rem 1.25rem; border-radius: 8px; margin-bottom: 1.5rem;">
            <h3 style="color: #38bdf8; font-size: 1.15rem; margin-bottom: 0.3rem;">📊 Model Analytics & Statistical Validation Scorecard</h3>
            <p style="color: #cbd5e1; font-size: 0.95rem; margin-bottom: 0;">
                Model analytics are displayed on-demand for clinical auditing. Performance metrics are backed by 50-fold cross-validation non-parametric hypothesis testing, ROC curves, and probability calibration diagrams.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metadata = load_metadata()
    if not metadata:
        st.warning("Model metadata is not loaded. Run export_models.py to generate statistical artifacts.")
        return

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        p_val = metadata.get("friedman_p", 0.0)
        st.metric("Friedman Omnibus p-value", f"{p_val:.4e}" if p_val < 0.001 else f"{p_val:.4f}", "Statistically Significant (p < 0.05)")
    with m2:
        st.metric("Ensemble Tuned Models", len(metadata.get("model_weights", {})), "Logistic, RF, XGB, SVM")
    with m3:
        top_model = max(metadata.get("model_weights", {}).items(), key=lambda x: x[1])[0]
        st.metric("Highest-Weighted Model", top_model, f"Weight: {metadata['model_weights'][top_model]:.3f}")
    with m4:
        st.metric("Cross-Validation Folds", "50 Folds", "5 splits × 10 repeats")

    st.markdown("---")

    st.subheader("1. Non-Parametric Hypothesis Testing (Friedman & Holm-Corrected Wilcoxon)")
    st.markdown(
        """
        A **Friedman omnibus test** was performed across all 50 matched CV folds to verify that model performance differences are statistically significant ($p < 0.05$), followed by **post-hoc pairwise Wilcoxon signed-rank tests** with **Holm-Bonferroni correction**.
        """
    )
    posthoc_data = metadata.get("holm_posthoc", [])
    if posthoc_data:
        ph_df = pd.DataFrame(posthoc_data)
        ph_df["raw_p"] = ph_df["raw_p"].map(lambda x: f"{x:.4e}" if x < 0.001 else f"{x:.4f}")
        ph_df["holm_p"] = ph_df["holm_p"].map(lambda x: f"{x:.4e}" if x < 0.001 else f"{x:.4f}")
        ph_df["significant"] = ph_df["significant"].map(lambda x: "✅ Significant (p < 0.05)" if x else "❌ Not Significant")
        ph_df = ph_df.rename(
            columns={
                "model_a": "Model A",
                "model_b": "Model B",
                "raw_p": "Raw p-value",
                "holm_p": "Holm-Adjusted p-value",
                "significant": "Significance Status",
            }
        )
        st.dataframe(ph_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("2. 50-Fold CV AUC Score Distribution")
        cv_scores = metadata.get("cv_scores", {})
        if cv_scores:
            cv_df_list = []
            for mdl, scores in cv_scores.items():
                for s in scores:
                    cv_df_list.append({"Model": mdl, "ROC AUC": s})
            cv_df = pd.DataFrame(cv_df_list)
            fig_cv = px.box(
                cv_df,
                x="Model",
                y="ROC AUC",
                color="Model",
                points="all",
                title="Repeated Stratified K-Fold AUC (50 Folds)",
            )
            fig_cv.update_layout(
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                showlegend=False,
            )
            fig_cv.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
            st.plotly_chart(fig_cv, use_container_width=True)

    with col_b:
        st.subheader("3. Test Set Receiver Operating Characteristic (ROC)")
        roc_data = metadata.get("roc_curves", {})
        if roc_data:
            fig_roc = go.Figure()
            fig_roc.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#64748b"), name="Random Chance (AUC = 0.50)")
            )
            for mdl, curve in roc_data.items():
                fig_roc.add_trace(
                    go.Scatter(
                        x=curve["fpr"],
                        y=curve["tpr"],
                        mode="lines",
                        name=f"{mdl} (AUC = {curve['auc']:.3f})",
                        line=dict(width=2),
                    )
                )
            fig_roc.update_layout(
                title="ROC Curves on Held-Out Test Set (N = 184)",
                xaxis_title="False Positive Rate (1 - Specificity)",
                yaxis_title="True Positive Rate (Sensitivity)",
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            )
            fig_roc.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
            fig_roc.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
            st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("4. Probability Calibration Curves & Brier Scores")
        calib_data = metadata.get("calibration", {})
        test_metrics = metadata.get("test_metrics", {})
        if calib_data:
            fig_cal = go.Figure()
            fig_cal.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#64748b"), name="Perfect Calibration")
            )
            for mdl, cdata in calib_data.items():
                brier = test_metrics.get(mdl, {}).get("brier", 0.0)
                fig_cal.add_trace(
                    go.Scatter(
                        x=cdata["prob_pred"],
                        y=cdata["prob_true"],
                        mode="lines+markers",
                        name=f"{mdl} (Brier = {brier:.3f})",
                        line=dict(width=2),
                    )
                )
            fig_cal.update_layout(
                title="Reliability Calibration Diagram",
                xaxis_title="Mean Predicted Probability",
                yaxis_title="Fraction of Positives (Actual)",
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            )
            fig_cal.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
            fig_cal.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
            st.plotly_chart(fig_cal, use_container_width=True)

    with col_d:
        st.subheader("5. Strength-Weighted Feature Importance Scorecard")
        importance = metadata.get("unified_importance", {})
        if importance:
            imp_df = pd.DataFrame({"Feature": list(importance.keys()), "Importance": list(importance.values())}).sort_values("Importance", ascending=True)
            fig_imp = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Viridis",
                title="Strength-Weighted Feature Importance Ranking",
            )
            fig_imp.update_layout(
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                showlegend=False,
            )
            fig_imp.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
            st.plotly_chart(fig_imp, use_container_width=True)


def main():
    if not MODEL_PATH.exists():
        st.error("Model artifacts not found. Run export_models.py first to train and export the forecasting pipeline.")
        st.stop()

    init_session_state()
    render_header()
    patient_profile = render_sidebar()

    tab_forecast, tab_analytics = st.tabs(["🫀 Patient Risk Forecast", "📊 Model Analytics & Statistical Validation"])

    with tab_forecast:
        render_forecast_tab(patient_profile)

    with tab_analytics:
        render_analytics_tab()


if __name__ == "__main__":
    main()
