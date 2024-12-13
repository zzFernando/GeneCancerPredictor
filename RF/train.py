import os
import joblib
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, confusion_matrix, matthews_corrcoef,
    mean_absolute_error, mean_squared_error, cohen_kappa_score, roc_curve
)
import numpy as np

# Configuração do dataset
dataset_url = "https://sbcb.inf.ufrgs.br/data/cumida/Genes/Liver/GSE14520_U133A/Liver_GSE14520_U133A.csv"
file_path = 'Liver_GSE14520_U133A.csv'
if not os.path.exists(file_path):
    urllib.request.urlretrieve(dataset_url, file_path)
dataset = pd.read_csv(file_path)

# Divisão dos dados
X, y = dataset.drop(['samples', 'type'], axis=1), dataset['type']
y_binary = y.map({'HCC': 1, 'normal': 0})
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Modelo
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', class_weight='balanced', random_state=42
)
model.fit(X_train_scaled, y_train)

# Treinamento no Conjunto Completo
y_train_pred = model.predict(X_train_scaled)
accuracy_train = accuracy_score(y_train, y_train_pred)
precision_train = precision_score(y_train, y_train_pred)
recall_train = recall_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)
mcc_train = matthews_corrcoef(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
conf_matrix_train = confusion_matrix(y_train, y_train_pred)

print("\n=== Treinamento no Conjunto Completo ===")
print(f"Acurácia: {accuracy_train:.4f}")
print(f"Precisão: {precision_train:.4f}")
print(f"Recall: {recall_train:.4f}")
print(f"F1-Score: {f1_train:.4f}")
print(f"MCC: {mcc_train:.4f}")
print(f"Mean Absolute Error: {mae_train:.4f}")
print(f"Root Mean Squared Error: {rmse_train:.4f}")
print(f"\nConfusion Matrix:\n{conf_matrix_train}")

# Validação Cruzada Estratificada
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
y_cv_pred = cross_val_predict(model, scaler.fit_transform(X), y_binary, cv=skf)

accuracy_cv = accuracy_score(y_binary, y_cv_pred)
precision_cv = precision_score(y_binary, y_cv_pred)
recall_cv = recall_score(y_binary, y_cv_pred)
f1_cv = f1_score(y_binary, y_cv_pred)
mcc_cv = matthews_corrcoef(y_binary, y_cv_pred)
kappa_cv = cohen_kappa_score(y_binary, y_cv_pred)
conf_matrix_cv = confusion_matrix(y_binary, y_cv_pred)
roc_auc_cv = roc_auc_score(y_binary, y_cv_pred)

print("\n=== Validação Cruzada Estratificada ===")
print(f"Acurácia: {accuracy_cv:.4f}")
print(f"Precisão: {precision_cv:.4f}")
print(f"Recall: {recall_cv:.4f}")
print(f"F1-Score: {f1_cv:.4f}")
print(f"MCC: {mcc_cv:.4f}")
print(f"Kappa: {kappa_cv:.4f}")
print(f"ROC AUC: {roc_auc_cv:.4f}")
print(f"\nConfusion Matrix:\n{conf_matrix_cv}")

# Detalhamento por Classe
report_cv = classification_report(y_binary, y_cv_pred, target_names=['Normal', 'HCC'])
print("\n=== Detalhamento por Classe ===")
print(report_cv)
