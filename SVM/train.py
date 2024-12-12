import os
import joblib
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

# Carrega o dataset
dataset = pd.read_csv(file_path)

# Define as variáveis independentes e dependentes
X = dataset.drop(['samples', 'type'], axis=1)
y = dataset['type']

# Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normalização dos dados
    ('classifier', SVC(random_state=42))
])

param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf'],  
    'classifier__gamma': ['scale', 'auto', 0.01, 0.1],
}

# Grid Search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)   

# Obtém os melhores parâmetros
best_params = grid_search.best_params_
print("Melhores parâmetros encontrados:", best_params)

# Faz previsões com o modelo ajustado
y_pred = grid_search.predict(X_test)

# Avalia o modelo no conjunto de teste com o threshold padrão
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia: {accuracy:.2f}")
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# Salva o melhor modelo treinado
joblib.dump(grid_search.best_estimator_, 'best_random_forest_model.pkl')
