import os
import joblib
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

# Define as variáveis independentes e dependentes
X = dataset.drop(['samples', 'type'], axis=1)  # Remove colunas não usadas como features
y = dataset['type']  # A coluna de rótulos (câncer ou não)

# Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define o pipeline de normalização e o modelo
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normalização dos dados
    ('classifier', RandomForestClassifier(random_state=42))
])

# Parâmetros para o Grid Search, reduzindo hiperparâmetros
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [5, 10],
    'classifier__min_samples_leaf': [2, 4],
    'classifier__max_features': ['sqrt'],
    'classifier__class_weight': ['balanced']
}

# Configura o GridSearchCV com o pipeline
grid_search = GridSearchCV(estimator=pipeline,
                           param_grid=param_grid,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=2)

# Executa o Grid Search nos dados de treino
print("Iniciando o Grid Search para otimização de hiperparâmetros...")
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

# Exibe as 10 features mais importantes se disponíveis
best_model = grid_search.best_estimator_.named_steps['classifier']
feature_importances = pd.DataFrame(best_model.feature_importances_, index=X.columns, columns=['Importance'])
print("\nAs 10 features mais importantes:")
print(feature_importances.sort_values(by='Importance', ascending=False).head(10))

# Salva o melhor modelo treinado
joblib.dump(grid_search.best_estimator_, 'best_random_forest_model.pkl')
print("Melhor modelo salvo como 'best_random_forest_model.pkl'.")
