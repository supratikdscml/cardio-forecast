"""One-time export: trains models from the notebook pipeline and saves artifacts for app.py."""
import os
import json
import pandas as pd
import numpy as np
import joblib
from itertools import combinations
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.model_selection import (train_test_split, RepeatedStratifiedKFold,
                                      cross_val_score, RandomizedSearchCV, StratifiedKFold)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, brier_score_loss, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve

os.makedirs("models", exist_ok=True)

df = pd.read_csv("heart.csv") if os.path.exists("heart.csv") else pd.read_csv(
    "https://raw.githubusercontent.com/anik199/Heart-failure-prediction/main/heart.csv")
df.to_csv("heart.csv", index=False)

df['RestingBP'] = df['RestingBP'].replace(0, np.nan)
df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']
X['Sex'] = X['Sex'].map({'M': 1, 'F': 0})
X['ExerciseAngina'] = X['ExerciseAngina'].map({'Y': 1, 'N': 0})

onehot_cols = ['ChestPainType', 'RestingECG', 'ST_Slope']
numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'Sex', 'ExerciseAngina']
preprocessor = ColumnTransformer([
    ('num', Pipeline([('scale', StandardScaler()), ('impute', KNNImputer(n_neighbors=5))]), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), onehot_cols)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
models_baseline = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric='auc', random_state=42, n_jobs=-1),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42)
}
cv_results = {}
for name, clf in models_baseline.items():
    pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=rskf, scoring='roc_auc', n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name}: {scores.mean():.3f}")

cv_tuner = StratifiedKFold(5, shuffle=True, random_state=42)
rf_search = RandomizedSearchCV(
    Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))]),
    {'clf__n_estimators': [100, 200, 300], 'clf__max_depth': [3, 4, 6, None], 'clf__min_samples_split': [2, 5, 10]},
    n_iter=15, scoring='roc_auc', cv=cv_tuner, random_state=42, n_jobs=-1)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

xgb_search = RandomizedSearchCV(
    Pipeline([('prep', preprocessor), ('clf', XGBClassifier(eval_metric='auc', random_state=42, n_jobs=-1))]),
    {'clf__n_estimators': [100, 200, 300], 'clf__max_depth': [2, 3, 4, 6], 'clf__learning_rate': [0.01, 0.05, 0.1]},
    n_iter=15, scoring='roc_auc', cv=cv_tuner, random_state=42, n_jobs=-1)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_

logreg_search = RandomizedSearchCV(
    Pipeline([('prep', preprocessor), ('clf', LogisticRegression(max_iter=1000, random_state=42))]),
    {'clf__C': [0.01, 0.1, 1, 10, 100]}, n_iter=5, scoring='roc_auc', cv=cv_tuner, random_state=42, n_jobs=-1)
logreg_search.fit(X_train, y_train)
best_logreg = logreg_search.best_estimator_

svm_search = RandomizedSearchCV(
    Pipeline([('prep', preprocessor), ('clf', SVC(probability=True, random_state=42))]),
    {'clf__C': [0.1, 1, 10, 100], 'clf__kernel': ['rbf', 'linear']},
    n_iter=8, scoring='roc_auc', cv=cv_tuner, random_state=42, n_jobs=-1)
svm_search.fit(X_train, y_train)
best_svm = svm_search.best_estimator_

model_order = list(cv_results.keys())
score_matrix = np.array([cv_results[m] for m in model_order])
friedman_stat, friedman_p = friedmanchisquare(*score_matrix)
pairs = list(combinations(model_order, 2))
raw_pvals = [wilcoxon(cv_results[a], cv_results[b])[1] for a, b in pairs]
order = np.argsort(raw_pvals)
m = len(raw_pvals)
holm_pvals = np.empty(m)
running_max = 0.0
for rank, idx in enumerate(order):
    adj = (m - rank) * raw_pvals[idx]
    running_max = max(running_max, adj)
    holm_pvals[idx] = min(running_max, 1.0)
posthoc_df = pd.DataFrame({
    'model_a': [p[0] for p in pairs], 'model_b': [p[1] for p in pairs],
    'raw_p': raw_pvals, 'holm_p': holm_pvals, 'significant': holm_pvals < 0.05
})

tuned_models = {'Logistic Regression': best_logreg, 'Random Forest': best_rf,
                'XGBoost': best_xgb, 'SVM (RBF)': best_svm}
brier_tuned, test_auc_tuned, test_metrics = {}, {}, {}
for name, mdl in tuned_models.items():
    proba = mdl.predict_proba(X_test)[:, 1]
    pred = mdl.predict(X_test)
    brier_tuned[name] = brier_score_loss(y_test, proba)
    test_auc_tuned[name] = roc_auc_score(y_test, proba)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    test_metrics[name] = {
        'auc': float(test_auc_tuned[name]), 'brier': float(brier_tuned[name]),
        'accuracy': float(accuracy_score(y_test, pred)),
        'precision': float(precision_score(y_test, pred)),
        'recall': float(recall_score(y_test, pred)),
        'f1': float(f1_score(y_test, pred)),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }

cv_bonus = {m: 0 for m in model_order}
if friedman_p < 0.05:
    for _, row in posthoc_df.iterrows():
        if row['significant']:
            a, b = row['model_a'], row['model_b']
            cv_bonus[a if cv_results[a].mean() > cv_results[b].mean() else b] += 1

strength_df = pd.DataFrame({'test_auc': test_auc_tuned, 'brier': brier_tuned, 'cv_significant_wins': cv_bonus})
strength_df['auc_norm'] = (strength_df['test_auc'] - strength_df['test_auc'].min()) / (strength_df['test_auc'].max() - strength_df['test_auc'].min() + 1e-9)
strength_df['calib_norm'] = 1 - (strength_df['brier'] - strength_df['brier'].min()) / (strength_df['brier'].max() - strength_df['brier'].min() + 1e-9)
max_wins = max(strength_df['cv_significant_wins'].max(), 1)
strength_df['sig_norm'] = strength_df['cv_significant_wins'] / max_wins
strength_df['composite'] = 0.5 * strength_df['auc_norm'] + 0.35 * strength_df['calib_norm'] + 0.15 * strength_df['sig_norm']
strength_df['weight'] = strength_df['composite'] / strength_df['composite'].sum()
model_weights = strength_df['weight'].to_dict()

prep = best_logreg.named_steps['prep']
cat_names = list(prep.named_transformers_['cat'].get_feature_names_out(onehot_cols))
feature_names = numeric_cols + cat_names
importance_results = {}
for name, mdl in tuned_models.items():
    clf_step = mdl.named_steps['clf']
    if hasattr(clf_step, 'feature_importances_'):
        imp = clf_step.feature_importances_
    elif hasattr(clf_step, 'coef_'):
        imp = clf_step.coef_[0]
    else:
        X_test_prepped = prep.transform(X_test)
        if hasattr(X_test_prepped, 'toarray'):
            X_test_prepped = X_test_prepped.toarray()
        perm = permutation_importance(clf_step, X_test_prepped, y_test, scoring='roc_auc', n_repeats=5, random_state=42, n_jobs=-1)
        imp = perm.importances_mean
    importance_results[name] = imp

norm_imp_df = pd.DataFrame({
    name: (np.abs(imp) - np.abs(imp).min()) / (np.abs(imp).max() - np.abs(imp).min() + 1e-9)
    for name, imp in importance_results.items()
}, index=feature_names)
weights_aligned = pd.Series({m: model_weights[m] for m in norm_imp_df.columns})
unified_importance = (norm_imp_df * weights_aligned).sum(axis=1)
unified_importance = (unified_importance / unified_importance.sum()).sort_values(ascending=False)

corr_df = pd.get_dummies(X, columns=onehot_cols, drop_first=False)
corr = corr_df.corr().fillna(0)
calibration_data = {}
for name, mdl in tuned_models.items():
    proba = mdl.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=5, strategy='uniform')
    calibration_data[name] = {'prob_true': prob_true.tolist(), 'prob_pred': prob_pred.tolist()}

joblib.dump({
    'models': tuned_models, 'weights': model_weights, 'unified_importance': unified_importance,
    'feature_names': feature_names, 'numeric_cols': numeric_cols, 'onehot_cols': onehot_cols
}, "models/forecaster_model.joblib")

with open("models/model_metadata.json", "w") as f:
    json.dump({
        'test_metrics': test_metrics, 'friedman_p': float(friedman_p),
        'holm_posthoc': posthoc_df.to_dict(orient='records'),
        'model_weights': {k: float(v) for k, v in model_weights.items()},
        'unified_importance': unified_importance.to_dict(),
        'correlation': {'columns': list(corr.columns), 'values': corr.values.tolist()},
        'calibration': calibration_data, 'y_test': y_test.tolist()
    }, f, indent=2)

print("Saved models/forecaster_model.joblib and models/model_metadata.json")
