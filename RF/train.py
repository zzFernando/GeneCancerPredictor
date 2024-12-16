import os
import json
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, mean_absolute_error,
    mean_squared_error, cohen_kappa_score, classification_report
)
from joblib import Parallel, delayed

dataset_url = "https://sbcb.inf.ufrgs.br/data/cumida/Genes/Liver/GSE14520_U133A/Liver_GSE14520_U133A.csv"
file_path = 'Liver_GSE14520_U133A.csv'
if not os.path.exists(file_path):
    urllib.request.urlretrieve(dataset_url, file_path)
dataset = pd.read_csv(file_path)

X, y = dataset.drop(['samples', 'type'], axis=1), dataset['type']
y_binary = y.map({'HCC': 1, 'normal': 0})

scaler = StandardScaler()

def process_rf(params, X, y, skf):
    y_cv_true, y_cv_pred = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

        X_train_cv_scaled = scaler.fit_transform(X_train_cv)
        X_test_cv_scaled = scaler.transform(X_test_cv)

        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            class_weight=params['class_weight'],
            random_state=42
        )

        model.fit(X_train_cv_scaled, y_train_cv)
        y_pred_fold = model.predict(X_test_cv_scaled)

        y_cv_true.extend(y_test_cv)
        y_cv_pred.extend(y_pred_fold)

    metrics = compute_metrics(np.array(y_cv_true), np.array(y_cv_pred))
    metrics["params"] = params
    return metrics

def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "Mean Absolute Error": mean_absolute_error(y_true, y_pred),
        "Root Mean Squared Error": np.sqrt(mean_squared_error(y_true, y_pred)),
        "Confusion Matrix": cm.tolist()
    }

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced']
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("Starting parallel grid search for Random Forest...")

results = Parallel(n_jobs=-1)(
    delayed(process_rf)(params, X, y_binary, skf)
    for params in ParameterGrid(param_grid)
)

best_result = max(results, key=lambda x: x["Recall"])

with open('best_rf_metrics.json', 'w') as f:
    json.dump(best_result, f, indent=4)

print("Best Random Forest metrics saved to best_rf_metrics.json")
