# Análise de Dados e Classificação com Random Forest

Este projeto realiza a análise de dados genômicos relacionados ao câncer de fígado utilizando um pipeline de Machine Learning. O objetivo é treinar e otimizar um modelo de Random Forest para prever o tipo de amostra (câncer ou não) com base nas features fornecidas.

## Estrutura do Projeto

O projeto utiliza diversas técnicas e ferramentas de Machine Learning, incluindo pré-processamento, seleção de hiperparâmetros, avaliação de modelos e visualização de resultados.

---

## Técnicas e Ferramentas Utilizadas

### **Linguagem de Programação**
- **Python**: A linguagem principal utilizada para a análise e construção do pipeline de Machine Learning.

### **Manipulação de Dados**
- **Pandas**: Para carregamento, limpeza e manipulação de dados.
- **NumPy**: Suporte a cálculos matemáticos e operações de arrays.

### **Pré-processamento**
- **StandardScaler (Sklearn)**: Para normalizar os dados, garantindo que todas as features tenham a mesma escala.

### **Modelagem**
- **Random Forest Classifier (Sklearn)**: Modelo de aprendizado supervisionado baseado em árvores de decisão.

### **Divisão dos Dados**
- **train_test_split (Sklearn)**: Divisão do dataset em conjunto de treino (80%) e teste (20%).

### **Otimização de Hiperparâmetros**
- **GridSearchCV (Sklearn)**: Busca dos melhores hiperparâmetros para o modelo por meio de validação cruzada.

### **Métricas de Avaliação**
- **accuracy_score (Sklearn)**: Acurácia do modelo.
- **classification_report (Sklearn)**: Relatório detalhado com métricas como precisão, recall e F1-score.
- **confusion_matrix (Sklearn)**: Matriz de confusão para avaliar previsões.

### **Visualização**
- **Matplotlib**: Para criação de gráficos e visualizações gerais.
- **Seaborn**: Para gráficos detalhados, como as importâncias das features.
- **ConfusionMatrixDisplay (Sklearn)**: Visualização da matriz de confusão.
- **learning_curve (Sklearn)**: Gráficos da curva de aprendizado.
- **validation_curve (Sklearn)**: Gráficos para avaliar o impacto de diferentes valores de hiperparâmetros.

### **Salvamento do Modelo**
- **Joblib**: Para salvar e carregar o modelo treinado.

### **Dados**
- **Dataset**: Os dados são genômicos e foram obtidos do [Banco de Dados CUMIDA](https://sbcb.inf.ufrgs.br/data/cumida/). O arquivo utilizado neste projeto é `Liver_GSE14520_U133A.csv`.

---

## Estrutura do Código

1. **Baixando o Dataset**  
   Verifica se o dataset está presente na máquina local. Caso contrário, faz o download do arquivo.

2. **Pré-processamento**  
   - Carrega o dataset e realiza a separação entre features (`X`) e rótulos (`y`).
   - Divide os dados em conjuntos de treino e teste.

3. **Pipeline de Machine Learning**  
   - Cria um pipeline para normalizar os dados e treinar o modelo de Random Forest.

4. **Otimização de Hiperparâmetros**  
   - Utiliza o `GridSearchCV` para encontrar os melhores parâmetros para o modelo.

5. **Avaliação do Modelo**  
   - Calcula métricas de desempenho no conjunto de teste.
   - Gera relatórios e gráficos, incluindo:
     - Matriz de confusão.
     - Importância das features.
     - Curva de aprendizado.
     - Curva de validação.

6. **Salvamento do Modelo**  
   - O melhor modelo ajustado é salvo no arquivo `best_random_forest_model.pkl`.

---

## Resultados e Visualizações

O código produz as seguintes visualizações e relatórios:

1. **Matriz de Confusão**  
   Exibe o desempenho do modelo na classificação das amostras.

2. **Importância das Features**  
   Lista e plota as 10 features mais importantes.

3. **Curva de Aprendizado**  
   Mostra como a acurácia varia com o tamanho do conjunto de treinamento.

4. **Curva de Validação**  
   Avalia o impacto de diferentes valores para o hiperparâmetro `n_estimators`.

---

## Como Executar

1. **Requisitos**  
   Instale as dependências necessárias com o seguinte comando:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn joblib
   ```

2. **Executar o Script**  
   Salve o código em um arquivo `.py` e execute com:
   ```bash
   python nome_do_arquivo.py
   ```

3. **Resultados**  
   - O melhor modelo será salvo como `best_random_forest_model.pkl`.
   - As visualizações serão exibidas diretamente.

---

## Referências

- [Documentação do Scikit-learn](https://scikit-learn.org/)
- [Dataset CUMIDA](https://sbcb.inf.ufrgs.br/data/cumida/)