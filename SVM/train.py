import os
import urllib.request
import numpy as np
import pandas as pd
import json
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
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

# Aplicar SMOTE para balancear as classes
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y_binary)

# Escala os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)

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
    n_iter=15,  # Limita para 15 combinações aleatórias
    cv=10,  # Validação cruzada com 10 folds
    scoring='f1',
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_scaled, y_balanced)
best_model = random_search.best_estimator_

print(f"Melhores parâmetros: {random_search.best_params_}")

# Avaliação em TODOS OS DADOS
y_pred = best_model.predict(X_scaled)
accuracy = accuracy_score(y_balanced, y_pred)
precision = precision_score(y_balanced, y_pred)
recall = recall_score(y_balanced, y_pred)
f1 = f1_score(y_balanced, y_pred)
mcc = matthews_corrcoef(y_balanced, y_pred)
mae = mean_absolute_error(y_balanced, y_pred)
rmse = np.sqrt(mean_squared_error(y_balanced, y_pred))
kappa = cohen_kappa_score(y_balanced, y_pred)
conf_matrix = confusion_matrix(y_balanced, y_pred).tolist()

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
