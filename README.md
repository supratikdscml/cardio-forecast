# 🫀 Cardio Forecast | Enterprise Clinical Decision & Risk Engine

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://cardio-forecast-djk7tmjscadddfkrjtgtmf.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-111111?style=for-the-badge&logo=xgboost)](https://xgboost.ai/)
[![Plotly](https://img.shields.io/badge/Plotly-6.9.0-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)

An enterprise-grade, transparent, and auditable cardiovascular risk assessment platform designed for clinical decision support. **Cardio Forecast** combines an ensemble of tuned machine learning classifiers with non-parametric hypothesis testing, probability calibration, interactive decision cutoff simulation, and personalized explainable AI (XAI).

🔗 **Live Deployed App**: [https://cardio-forecast-djk7tmjscadddfkrjtgtmf.streamlit.app/](https://cardio-forecast-djk7tmjscadddfkrjtgtmf.streamlit.app/)

---

## 🌟 Executive Summary & Key Highlights

Unlike simple single-model classifiers, **Cardio Forecast** implements a **statistically grounded forecasting engine**:
- **Strength-Weighted Multi-Model Ensemble**: Combines tuned Logistic Regression, Random Forest, XGBoost, and Support Vector Machines (SVM RBF).
- **Non-Parametric Hypothesis Testing**: Model weighting is derived from a 50-fold repeated stratified cross-validation benchmark ($5 \text{ splits} \times 10 \text{ repeats}$), evaluated using a **Friedman omnibus test** ($p = 2.47 \times 10^{-8}$) and **Holm-corrected post-hoc Wilcoxon signed-rank tests**.
- **Personalized Explainable AI (XAI)**: Ranks patient-specific risk drivers by combining strength-weighted global feature importance with local signed linear effect contributions.
- **Cost-Sensitive Decision Simulator**: Allows clinicians to adjust False Negative vs. False Positive penalty ratios to determine cost-optimal decision cutoffs.
- **Zero Fabricated Data**: Built strictly on authentic clinical features (`heart.csv` dataset schema) and validated mathematical formulas.

---

## 🚀 Interactive Application Architecture

The application is structured into two core operational views:

### 1. 🫀 Patient Risk Forecast (Primary Decision Engine)
- **Real-Time Reactive Intake**: Instant recalculation of risk scores, gauge visualizations, and driver impact on any slider or dropdown change without manual form submission locks.
- **Clinical Preset Loader**: Load pre-configured clinical archetypes (*Demo Patient: Asymptomatic High-Risk*, *Low Risk Baseline*, *Severe Ischemia Suspect*, *Atypical Female Profile*).
- **Plotly Interactive Gauge**: Color-coded risk score visualization ($0 - 100\%$) categorized into four clinical bands:
  - 🟢 **Low Risk** ($0 - 25\%$) — Routine preventive monitoring.
  - 🟡 **Moderate Risk** ($25 - 50\%$) — Lifestyle interventions & regular monitoring.
  - 🟠 **High Risk** ($50 - 75\%$) — Diagnostic evaluation (stress ECG / echocardiogram).
  - 🔴 **Very High Risk** ($75 - 100\%$) — Immediate clinical consultation & targeted therapeutic intervention.
- **Ensemble Model Consensus**: Side-by-side comparison of individual model probabilities and strength weights.
- **Top Personal Risk Drivers**: Color-coded horizontal bar chart (Red: Increases Risk, Green: Lowers Risk) highlighting patient-specific ischemia factors.
- **Interactive Threshold Cutoff Simulator**: Adjust False Negative penalty multipliers ($1.0\times - 10.0\times$) to optimize clinical decision thresholds.

### 2. 📊 Model Analytics & Statistical Validation (On-Demand Auditing)
- **Friedman Omnibus Test & Holm Post-Hoc Matrix**: Complete statistical significance table for cross-model comparisons.
- **50-Fold Repeated CV Boxplots**: Score distributions across 50 cross-validation folds.
- **Held-Out Test Set ROC Curves**: Interactive Plotly ROC curves annotated with test set AUC scores ($N = 184$).
- **Reliability Calibration Diagram**: Probability calibration curves and Brier score metrics.
- **Strength-Weighted Feature Importance Scorecard**: Global feature ranking weighted by model trustworthiness.

---

## 📋 Clinical Feature Dictionary & Reference Ranges

| Feature | Type | Reference / Range | Clinical Description |
| :--- | :--- | :--- | :--- |
| **Age** | Numeric | 28 – 77 years | Patient age in years. |
| **Sex** | Categorical | `M`, `F` | Biological sex (`M`: Male, `F`: Female). |
| **ChestPainType** | Categorical | `TA`, `ATA`, `NAP`, `ASY` | `TA`: Typical Angina, `ATA`: Atypical Angina, `NAP`: Non-Anginal Pain, `ASY`: Asymptomatic. |
| **RestingBP** | Numeric | 90 – 120 mm Hg | Resting blood pressure on admission. Biologically impossible zeros auto-imputed via KNN. |
| **Cholesterol** | Numeric | 125 – 200 mg/dl | Serum cholesterol level. Zeros auto-imputed via KNN. |
| **FastingBS** | Binary | `0`: ≤ 120, `1`: > 120 mg/dl | Fasting blood sugar > 120 mg/dl indicates diabetic hyperglycemia. |
| **RestingECG** | Categorical | `Normal`, `ST`, `LVH` | `Normal`: Normal, `ST`: ST-T wave abnormality, `LVH`: Left ventricular hypertrophy. |
| **MaxHR** | Numeric | 60 – 200 bpm | Maximum heart rate achieved during exercise stress testing. |
| **ExerciseAngina** | Binary | `N` (No), `Y` (Yes) | Exercise-induced angina. `Y` strongly correlates with myocardial ischemia. |
| **Oldpeak** | Numeric | 0.0 – 1.0 mm | ST depression induced by exercise relative to rest. Elevated values signal severe ischemia. |
| **ST_Slope** | Categorical | `Up`, `Flat`, `Down` | Slope of peak exercise ST segment. `Up` is protective; `Flat` and `Down` indicate severe CAD risk. |

---

## 🛠️ Project File Structure

```text
cardio-forecast/
├── app.py                                  # Enterprise Streamlit application & interactive UI
├── export_models.py                        # Model training, hyperparameter tuning & statistical metadata exporter
├── HD_Prediction_Forecasting_Engine.ipynb  # Jupyter notebook containing exploratory analysis & validation
├── heart.csv                               # Clinical heart disease dataset
├── requirements.txt                        # Python dependencies
├── pyrightconfig.json                      # IDE language server configuration
├── .vscode/                                # VS Code workspace interpreter settings
└── models/
    ├── forecaster_model.joblib             # Serialized ensemble model pipeline
    └── model_metadata.json                 # Exported statistical test matrices, ROC & calibration curves
```

---

## 💻 Local Installation & Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/supratikdscml/cardio-forecast.git
cd cardio-forecast
```

### 2. Create and Activate a Virtual Environment
**Windows (PowerShell)**:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**macOS / Linux**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Export Model Artifacts (Optional)
If model artifacts need to be re-generated from scratch:
```bash
python export_models.py
```

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔬 Statistical Methodology

1. **Preprocessing**: Biologically invalid zeroes in `RestingBP` and `Cholesterol` are treated as missing values and imputed using $K$-Nearest Neighbors ($K=5$) within a scikit-learn `Pipeline`.
2. **Repeated Cross-Validation**: Baseline models undergo $5 \times 10$ Repeated Stratified K-Fold CV to generate paired distribution samples.
3. **Hypothesis Testing**:
   - **Friedman Test**: Tests the null hypothesis ($H_0$) that all models exhibit equal performance across folds.
   - **Wilcoxon Signed-Rank Test**: Pairwise non-parametric comparisons with **Holm-Bonferroni correction** to control Family-Wise Error Rate (FWER).
4. **Model Composite Weighting**:
   $$\text{Composite Weight} = 0.50 \times \text{AUC}_{\text{norm}} + 0.35 \times \text{Calibration}_{\text{norm}} + 0.15 \times \text{HypothesisWins}_{\text{norm}}$$

---

## ⚖️ Clinical & Ethical Disclaimer
*Cardio Forecast is developed for educational, statistical research, and decision-support demonstration purposes only. It does not replace clinical judgment, diagnostic imaging, or professional medical consultation.*

---
**Maintained by**: [Supratik Ghosh](https://github.com/supratikdscml)
