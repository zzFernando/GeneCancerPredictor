import os
import urllib.request
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# URLs e caminhos dos arquivos
new_dataset_url = "https://sbcb.inf.ufrgs.br/data/cumida/Genes/Liver/GSE14520_U133_2/Liver_GSE14520_U133_2.csv"
new_file_path = 'Liver_GSE14520_U133_2.csv'

# Baixa o novo dataset se não estiver no diretório
if not os.path.exists(new_file_path):
    print("Baixando o novo dataset...")
    urllib.request.urlretrieve(new_dataset_url, new_file_path)
    print("Novo dataset baixado com sucesso.")
else:
    print("Novo dataset já existe na pasta.")

# Carrega o novo dataset e o modelo treinado
new_dataset = pd.read_csv(new_file_path)
model = joblib.load('best_random_forest_model.pkl')
print("Modelo carregado com sucesso.")

# Prepara os dados para classificação e avaliação
X_new = new_dataset.drop(['samples', 'type'], axis=1)  # Features do novo dataset
y_true = new_dataset['type']  # Rótulos reais para avaliação

# Classifica usando o modelo treinado
y_pred = model.predict(X_new)

# Avalia a precisão do modelo
accuracy = (y_pred == y_true).mean()
report = classification_report(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

# Exibe os resultados
print(f"\nAcurácia no novo dataset: {accuracy:.2f}")
print("\nRelatório de Classificação:")
print(report)
print("\nMatriz de Confusão:")
print(conf_matrix)
