import os
import json
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, mean_absolute_error,
    mean_squared_error, cohen_kappa_score, classification_report
)
from joblib import Parallel, delayed

# Configuração do caminho do dataset
dataset_url = "https://sbcb.inf.ufrgs.br/data/cumida/Genes/Liver/GSE14520_U133A/Liver_GSE14520_U133A.csv"
file_path = 'Liver_GSE14520_U133A.csv'

# Verifica se o dataset já foi baixado
if not os.path.exists(file_path):
    print("Baixando o dataset...")
    urllib.request.urlretrieve(dataset_url, file_path)
    print("Dataset baixado com sucesso.")
else:
    print("Dataset já existe na pasta.")

# Carrega o dataset
dataset = pd.read_csv(file_path)
X, y = dataset.drop(['samples', 'type'], axis=1), dataset['type']
y_binary = y.map({'HCC': 1, 'normal': 0})

scaler = StandardScaler()

# Função para processar uma combinação de parâmetros
def process_svm(params, X, y, skf):
    y_cv_true, y_cv_pred = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(C=params['C'], gamma=params['gamma'], kernel='rbf', probability=True, random_state=42))
        ])

        pipeline.fit(X_train_cv, y_train_cv)
        y_pred_fold = pipeline.predict(X_test_cv)

        y_cv_true.extend(y_test_cv)
        y_cv_pred.extend(y_pred_fold)

    metrics = compute_metrics(np.array(y_cv_true), np.array(y_cv_pred))
    metrics["params"] = params
    return metrics

# Função para calcular métricas
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

# Configuração da busca em grade
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1]
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("Starting parallel grid search for SVM...")

# Execução paralela do grid search
results = Parallel(n_jobs=-1)(
    delayed(process_svm)(params, X, y_binary, skf)
    for params in ParameterGrid(param_grid)
)

# Salva apenas o melhor resultado baseado em Recall
best_result = max(results, key=lambda x: x["Recall"])

with open('best_svm_metrics.json', 'w') as f:
    json.dump(best_result, f, indent=4)

print("Best SVM metrics saved to best_svm_metrics.json")
