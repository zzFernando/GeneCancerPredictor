import os
import json
import urllib.request
import pandas as pd
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, mean_absolute_error,
    mean_squared_error, cohen_kappa_score, classification_report
)
import numpy as np
from joblib import Parallel, delayed

# Dataset configuration
dataset_url = "https://sbcb.inf.ufrgs.br/data/cumida/Genes/Liver/GSE14520_U133A/Liver_GSE14520_U133A.csv"
file_path = 'Liver_GSE14520_U133A.csv'
if not os.path.exists(file_path):
    urllib.request.urlretrieve(dataset_url, file_path)
dataset = pd.read_csv(file_path)

X, y = dataset.drop(['samples', 'type'], axis=1), dataset['type']
y_binary = y.map({'HCC': 1, 'normal': 0})
scaler = StandardScaler()

def process_kmeans(params, X, y, skf, scaler):
    y_cv_true, y_cv_pred = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

        X_train_cv_scaled = scaler.fit_transform(X_train_cv)
        X_test_cv_scaled = scaler.transform(X_test_cv)

        kmeans = KMeans(
            n_clusters=2,
            init=params['init'],
            n_init=params['n_init'],
            max_iter=params['max_iter'],
            random_state=42
        )
        kmeans.fit(X_train_cv_scaled)

        cluster_to_class_map = {
            cluster: int(y_train_cv[kmeans.labels_ == cluster].mode().iloc[0])
            for cluster in np.unique(kmeans.labels_)
        }

        y_pred_fold = [
            cluster_to_class_map[label] for label in kmeans.predict(X_test_cv_scaled)
        ]

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
    'init': ['k-means++', 'random'],
    'n_init': [10, 20],
    'max_iter': [300, 500]
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("Starting parallel grid search for K-Means...")

results = Parallel(n_jobs=-1)(
    delayed(process_kmeans)(params, X, y_binary, skf, scaler)
    for params in ParameterGrid(param_grid)
)

# Save only the best result based on Recall
best_result = max(results, key=lambda x: x["Recall"])

with open('best_kmeans_metrics.json', 'w') as f:
    json.dump(best_result, f, indent=4)

print("Best K-Means metrics saved to best_kmeans_metrics.json")
