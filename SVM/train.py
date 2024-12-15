import os
import urllib.request
import numpy as np
import pandas as pd
import json
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, mean_absolute_error,
    mean_squared_error, cohen_kappa_score
)

# Configuração do dataset
dataset_url = "https://sbcb.inf.ufrgs.br/data/cumida/Genes/Liver/GSE14520_U133A/Liver_GSE14520_U133A.csv"
file_path = 'Liver_GSE14520_U133A.csv'

if not os.path.exists(file_path):
    urllib.request.urlretrieve(dataset_url, file_path)

dataset = pd.read_csv(file_path)
X, y = dataset.drop(['samples', 'type'], axis=1), dataset['type']
y_binary = y.map({'HCC': 1, 'normal': 0})

# Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Aplicar SMOTE apenas nos dados de treino
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Escala os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.fit_transform(X)  # Escala todo o dataset

# Randomized Search para SVM
param_dist = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 'scale'],
    'kernel': ['rbf'],
    'class_weight': ['balanced']
}

random_search = RandomizedSearchCV(
    estimator=SVC(random_state=42),
    param_distributions=param_dist,
    n_iter=15,
    cv=10,
    scoring='recall',
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train_scaled, y_train_balanced)
best_model = random_search.best_estimator_

print(f"Melhores parâmetros: {random_search.best_params_}")

# Avaliação no dataset completo
y_all_pred = best_model.predict(X_scaled)
accuracy = accuracy_score(y_binary, y_all_pred)
precision = precision_score(y_binary, y_all_pred)
recall = recall_score(y_binary, y_all_pred)
f1 = f1_score(y_binary, y_all_pred)
mcc = matthews_corrcoef(y_binary, y_all_pred)
mae = mean_absolute_error(y_binary, y_all_pred)
rmse = np.sqrt(mean_squared_error(y_binary, y_all_pred))
kappa = cohen_kappa_score(y_binary, y_all_pred)
conf_matrix = confusion_matrix(y_binary, y_all_pred).tolist()

# Salvar resultados em um arquivo JSON
result = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1,
    "MCC": mcc,
    "Kappa": kappa,
    "Mean Absolute Error": mae,
    "Root Mean Squared Error": rmse,
    "Confusion Matrix": conf_matrix,
    "params": random_search.best_params_
}

output_file = "best_svm_metrics.json"
with open(output_file, "w") as f:
    json.dump(result, f, indent=4)

print(f"Resultados salvos em {output_file}")
