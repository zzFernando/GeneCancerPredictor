import os
import urllib.request
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, confusion_matrix, matthews_corrcoef,
    mean_absolute_error, mean_squared_error, cohen_kappa_score
)

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

# Define as variáveis independentes e dependentes
X, y = dataset.drop(['samples', 'type'], axis=1), dataset['type']
y_binary = y.map({'HCC': 1, 'normal': 0})

# Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Configura o pipeline com StandardScaler, PCA e KNN
pipeline = Pipeline([
    ('scaler', StandardScaler()), # Escalona os dados
    ('pca', PCA(n_components=10)), # Reduz para 10 componentes principais
    ('knn', KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean'))
])

# Treina o modelo no conjunto de treinamento
pipeline.fit(X_train, y_train)

# Avaliação no conjunto de treinamento
y_train_pred = pipeline.predict(X_train)
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

# Stratified Cross-Validation com o pipeline
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
y_cv_pred = cross_val_predict(pipeline, X, y_binary, cv=skf)

accuracy_cv = accuracy_score(y_binary, y_cv_pred)
precision_cv = precision_score(y_binary, y_cv_pred)
recall_cv = recall_score(y_binary, y_cv_pred)
f1_cv = f1_score(y_binary, y_cv_pred)
mcc_cv = matthews_corrcoef(y_binary, y_cv_pred)
kappa_cv = cohen_kappa_score(y_binary, y_cv_pred)
roc_auc_cv = roc_auc_score(y_binary, y_cv_pred)
conf_matrix_cv = confusion_matrix(y_binary, y_cv_pred)

print("\n=== Stratified Cross-Validation ===")
print(f"Accuracy: {accuracy_cv:.4f}")
print(f"Precision: {precision_cv:.4f}")
print(f"Recall: {recall_cv:.4f}")
print(f"F1-Score: {f1_cv:.4f}")
print(f"MCC: {mcc_cv:.4f}")
print(f"Kappa: {kappa_cv:.4f}")
print(f"ROC AUC: {roc_auc_cv:.4f}")
print(f"\nConfusion Matrix:\n{conf_matrix_cv}")

# Class-Specific Performance
report_cv = classification_report(y_binary, y_cv_pred, target_names=['Normal', 'HCC'])
print("\n=== Class-Specific Performance ===")
print(report_cv)