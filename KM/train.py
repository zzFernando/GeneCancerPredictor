import os
import joblib
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, matthews_corrcoef,
    mean_absolute_error, mean_squared_error, cohen_kappa_score
)
import numpy as np

# Dataset configuration
dataset_url = "https://sbcb.inf.ufrgs.br/data/cumida/Genes/Liver/GSE14520_U133A/Liver_GSE14520_U133A.csv"
file_path = 'Liver_GSE14520_U133A.csv'
if not os.path.exists(file_path):
    urllib.request.urlretrieve(dataset_url, file_path)
dataset = pd.read_csv(file_path)

# Data split
X, y = dataset.drop(['samples', 'type'], axis=1), dataset['type']
y_binary = y.map({'HCC': 1, 'normal': 0})
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Optimized K-Means Model
kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42, max_iter=300)
kmeans.fit(X_train_scaled)

# Map clusters to labels
cluster_to_class_map = {
    cluster: int(y_train[kmeans.labels_ == cluster].mode().iloc[0]) for cluster in np.unique(kmeans.labels_)
}
y_train_pred = [cluster_to_class_map[label] for label in kmeans.labels_]

# Training metrics
accuracy_train = accuracy_score(y_train, y_train_pred)
precision_train = precision_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)
mcc_train = matthews_corrcoef(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
conf_matrix_train = confusion_matrix(y_train, y_train_pred)

print("\n=== Training on the Complete Dataset ===")
print(f"Accuracy: {accuracy_train:.4f}")
print(f"Precision: {precision_train:.4f}")
print(f"Recall: {recall_train:.4f}")
print(f"F1-Score: {f1_train:.4f}")
print(f"MCC: {mcc_train:.4f}")
print(f"Mean Absolute Error: {mae_train:.4f}")
print(f"Root Mean Squared Error: {rmse_train:.4f}")
print(f"\nConfusion Matrix:\n{conf_matrix_train}")

# Stratified Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
y_cv_pred = []
for train_idx, test_idx in skf.split(X, y_binary):
    kmeans.fit(scaler.fit_transform(X.iloc[train_idx]))
    cluster_to_class_map_cv = {
        cluster: int(y_binary.iloc[train_idx][kmeans.labels_ == cluster].mode().iloc[0])
        for cluster in np.unique(kmeans.labels_)
    }
    y_cv_pred.extend([cluster_to_class_map_cv[label] for label in kmeans.predict(scaler.transform(X.iloc[test_idx]))])

# Cross-validation metrics
accuracy_cv = accuracy_score(y_binary, y_cv_pred)
precision_cv = precision_score(y_binary, y_cv_pred)
recall_cv = recall_score(y_binary, y_cv_pred)
f1_cv = f1_score(y_binary, y_cv_pred)
mcc_cv = matthews_corrcoef(y_binary, y_cv_pred)
kappa_cv = cohen_kappa_score(y_binary, y_cv_pred)
conf_matrix_cv = confusion_matrix(y_binary, y_cv_pred)

print("\n=== Stratified Cross-Validation ===")
print(f"Accuracy: {accuracy_cv:.4f}")
print(f"Precision: {precision_cv:.4f}")
print(f"Recall: {recall_cv:.4f}")
print(f"F1-Score: {f1_cv:.4f}")
print(f"MCC: {mcc_cv:.4f}")
print(f"Kappa: {kappa_cv:.4f}")
print(f"\nConfusion Matrix:\n{conf_matrix_cv}")

# Class-Specific Performance
report_cv = classification_report(y_binary, y_cv_pred, target_names=['Normal', 'HCC'])
print("\n=== Class-Specific Performance ===")
print(report_cv)
