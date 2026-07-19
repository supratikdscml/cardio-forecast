"""
HD Prediction Forecasting Engine — Web UI
Flask server wrapping the HeartRiskForecaster from HD_Prediction_Forecasting_Engine.ipynb
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory


class HeartRiskForecaster:
    """
    Strength-weighted ensemble heart disease risk forecaster.
    Mirrors the HeartRiskForecaster class in HD_Prediction_Forecasting_Engine.ipynb.
    """

    RAW_COLUMNS = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                   'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
                   'Oldpeak', 'ST_Slope']

    RISK_BANDS = [
        (0.00, 0.25, 'Low'),
        (0.25, 0.50, 'Moderate'),
        (0.50, 0.75, 'High'),
        (0.75, 1.01, 'Very High'),
    ]

    def __init__(self, models: dict, weights: dict, unified_importance: pd.Series,
                 logreg_model, feature_names: list, numeric_cols: list, onehot_cols: list):
        self.models = models
        self.weights = weights
        self.unified_importance = unified_importance
        self.logreg_model = logreg_model
        self.feature_names = feature_names
        self.numeric_cols = numeric_cols
        self.onehot_cols = onehot_cols

    def _to_frame(self, patient: dict) -> pd.DataFrame:
        missing = [c for c in self.RAW_COLUMNS if c not in patient]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        row = {k: patient[k] for k in self.RAW_COLUMNS}
        df = pd.DataFrame([row])
        df['Sex'] = df['Sex'].map({'M': 1, 'F': 0}) if not pd.api.types.is_numeric_dtype(df['Sex']) else df['Sex']
        df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0}) if not pd.api.types.is_numeric_dtype(df['ExerciseAngina']) else df['ExerciseAngina']
        df['RestingBP'] = df['RestingBP'].replace(0, np.nan)
        df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
        return df

    def _risk_band(self, score: float) -> str:
        for lo, hi, label in self.RISK_BANDS:
            if lo <= score < hi:
                return label
        return self.RISK_BANDS[-1][2]

    def _local_explanation(self, patient_df: pd.DataFrame, top_n: int = 5):
        prep = self.logreg_model.named_steps['prep']
        clf = self.logreg_model.named_steps['clf']
        X_transformed = prep.transform(patient_df)
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()

        contrib = X_transformed[0] * clf.coef_[0]
        contrib_df = pd.DataFrame({'feature': self.feature_names, 'local_effect': contrib})
        contrib_df = contrib_df.set_index('feature')
        contrib_df['global_importance'] = self.unified_importance.reindex(contrib_df.index).fillna(0)
        contrib_df['driver_score'] = contrib_df['global_importance'] * contrib_df['local_effect'].abs()
        contrib_df['direction'] = np.where(contrib_df['local_effect'] > 0, 'increases risk', 'lowers risk')
        top = contrib_df.sort_values('driver_score', ascending=False).head(top_n)

        drivers_list = []
        for feature, row in top.iterrows():
            drivers_list.append({
                'feature': feature,
                'direction': row['direction'],
                'local_effect': float(row['local_effect']),
                'global_importance': float(row['global_importance']),
                'driver_score': float(row['driver_score'])
            })
        return drivers_list

    def predict(self, patient: dict) -> dict:
        patient_df = self._to_frame(patient)

        per_model = {}
        weighted_sum = 0.0
        for name, mdl in self.models.items():
            proba = mdl.predict_proba(patient_df)[:, 1][0]
            per_model[name] = float(proba)
            weighted_sum += proba * self.weights[name]

        risk_score = float(weighted_sum)
        band = self._risk_band(risk_score)
        drivers = self._local_explanation(patient_df)

        return {
            'risk_score': risk_score,
            'risk_percentage': risk_score * 100,
            'risk_band': band,
            'per_model_probability': per_model,
            'model_weights_used': {k: float(v) for k, v in self.weights.items()},
            'top_personal_drivers': drivers
        }


app = Flask(__name__, static_folder='static')
forecaster = None


def load_forecaster():
    global forecaster
    model_path = "models/forecaster_model.joblib"
    if not os.path.exists(model_path):
        print("Model not found. Run the forecasting-engine cells in HD_Prediction_Forecasting_Engine.ipynb to train and export models.")
        return False
    try:
        payload = joblib.load(model_path)
        forecaster = HeartRiskForecaster(
            models=payload['models'],
            weights=payload['weights'],
            unified_importance=payload['unified_importance'],
            logreg_model=payload['models']['Logistic Regression'],
            feature_names=payload['feature_names'],
            numeric_cols=payload['numeric_cols'],
            onehot_cols=payload['onehot_cols']
        )
        print("HeartRiskForecaster loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


load_forecaster()


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


@app.route('/api/predict', methods=['POST'])
def predict():
    global forecaster
    if forecaster is None and not load_forecaster():
        return jsonify({
            "status": "error",
            "message": "Trained model not found. Run the notebook forecasting-engine section first."
        }), 503

    try:
        data = request.json
        patient = {
            'Age': int(data['Age']),
            'Sex': str(data['Sex']),
            'ChestPainType': str(data['ChestPainType']),
            'RestingBP': int(data['RestingBP']),
            'Cholesterol': int(data['Cholesterol']),
            'FastingBS': int(data['FastingBS']),
            'RestingECG': str(data['RestingECG']),
            'MaxHR': int(data['MaxHR']),
            'ExerciseAngina': str(data['ExerciseAngina']),
            'Oldpeak': float(data['Oldpeak']),
            'ST_Slope': str(data['ST_Slope'])
        }
        result = forecaster.predict(patient)
        return jsonify({"status": "success", "prediction": result})
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    path = "models/model_metadata.json"
    if not os.path.exists(path):
        return jsonify({
            "status": "error",
            "message": "Model metadata not found. Train models via the notebook first."
        }), 503
    with open(path, 'r') as f:
        metadata = json.load(f)
    return jsonify({"status": "success", "metadata": metadata})


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
