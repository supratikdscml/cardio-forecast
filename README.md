# cardio-forecast

Cardio Forecast is a heart-disease risk prediction web app that combines multiple machine-learning models into a strength-weighted ensemble. The model comparison is backed by non-parametric statistical testing (Friedman test and post-hoc Wilcoxon signed-rank tests) so the forecast is grounded in model performance rather than a single arbitrary classifier.

## Features
- Web-based frontend for entering patient clinical data
- Strength-weighted ensemble risk scoring
- Per-model probability breakdown and ensemble weighting
- Personalized risk-driver explanation
- Model analytics tab with feature importance, calibration, and statistical comparisons

## Project Structure
- app.py: Flask backend serving the UI and prediction API
- export_models.py: Trains and exports the model artifacts used by the app
- static/: Frontend HTML, CSS, and JavaScript
- models/: Saved model and metadata files used at runtime
- requirements.txt: Python dependencies
- HD_Prediction_Forecasting_Engine.ipynb: Notebook containing the original analysis and forecasting engine logic

## Requirements
Python 3.9+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train and export the model artifacts
If the model files are missing from the models directory, run:

```bash
python export_models.py
```

This script trains the models, computes the ensemble weights and metadata, and saves the files required by the Flask app.

## Run the app locally
From the project root:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

## How to start the frontend from the GitHub repo
1. Clone the repository:
   ```bash
   git clone https://github.com/supratikdscml/cardio-forecast.git
   cd cardio-forecast
   ```
2. Create and activate a Python environment (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Train/export the model files if needed:
   ```bash
   python export_models.py
   ```
5. Start the backend/frontend server:
   ```bash
   python app.py
   ```
6. Open the browser at:
   ```text
   http://127.0.0.1:5000/
   ```

## Notes
- The frontend is served by Flask from the static directory.
- The app is intended for research/educational use and is not a clinical diagnostic tool.
- If you want to modify the UI, edit the files in the static directory.
