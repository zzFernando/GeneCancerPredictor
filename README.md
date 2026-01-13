# 🧬 Gene Cancer Predictor - Sistema de Predição de Câncer Hepático

Um projeto completo de **machine learning** para predição de **Hepatocarcinoma (HCC)** - câncer de fígado - utilizando dados de **expressão gênica** da base de dados **CuMiDa**. O projeto implementa e compara **6 modelos diferentes** de aprendizado de máquina com otimização de hiperparâmetros e análise detalhada de desempenho.

## 📋 Sumário

- [Visão Geral](#-visão-geral)
- [Conjunto de Dados](#-conjunto-de-dados)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Modelos Implementados](#-modelos-implementados)
- [Instalação](#-instalação)
- [Como Usar](#-como-usar)
- [Resultados e Comparação](#-resultados-e-comparação)
- [Visualizações](#-visualizações)
- [Requisitos e Dependências](#-requisitos-e-dependências)
- [Contribuições](#-contribuições)
- [Licença](#-licença)

---

## 🎯 Visão Geral

Este projeto aplica **técnicas avançadas de machine learning** para classificar amostras de tecido hepático como **HCC (cancerígeno)** ou **normal** com base em dados de **expressão gênica**. O objetivo principal é:

1. **Treinar e otimizar** 6 modelos diferentes de ML
2. **Comparar desempenho** entre os modelos
3. **Fornecer uma interface interativa** (Streamlit) para visualizar resultados
4. **Auxiliar na pesquisa médica** através de predições baseadas em dados genômicos

### Modelos Implementados:
- ✅ **FNN** (Feedforward Neural Network)
- ✅ **KNN** (K-Nearest Neighbors)
- ✅ **KMeans** (K-Means Clustering)
- ✅ **Naive Bayes** (Gaussian Naive Bayes)
- ✅ **Random Forest**
- ✅ **SVM** (Support Vector Machine)

---

## 📊 Conjunto de Dados

### Informações Gerais
- **Fonte**: [CuMiDa Database](https://sbcb.inf.ufrgs.br/cumida)
- **Dataset**: GSE14520_U133A (Liver)
- **Formato**: CSV com mais de 22.000 genes
- **Tamanho**: 357 amostras
- **Balanceamento**: Dados desbalanceados (mais amostras normais que HCC)

### Características do Dataset
- **Genes**: 22.278 características (expressão gênica)
- **Amostras**: 357 (HCC e normal)
- **Classes**: Binária (HCC = 1, Normal = 0)
- **Download Automático**: O sistema baixa o dataset automaticamente se não estiver presente

### Pré-processamento
1. **Normalização**: StandardScaler para padronizar valores
2. **Balanceamento**: SMOTE utilizado em alguns modelos (SVM)
3. **Redução de Dimensionalidade**: PCA aplicado onde necessário (KNN)
4. **Validação Cruzada**: 10-fold cross-validation em todos os modelos

---

## 📁 Estrutura do Projeto

```
GeneCancerPredictor/
├── app.py                          # Interface Streamlit (visualizações interativas)
├── requirements.txt                # Dependências principais
├── README.md                       # Este arquivo
├── LICENSE                         # Licença do projeto
│
├── FNN/                            # Feedforward Neural Network
│   ├── train.py                    # Script de treinamento
│   ├── best_fnn_metrics.json      # Métricas do melhor modelo
│   ├── requirements.txt            # Dependências específicas (PyTorch)
│   └── README.md                   # Documentação detalhada
│
├── KNN/                            # K-Nearest Neighbors
│   ├── train.py                    # Script de treinamento
│   ├── best_knn_metrics.json      # Métricas do melhor modelo
│   └── README.md                   # Documentação detalhada
│
├── KM/                             # K-Means Clustering
│   ├── train.py                    # Script de treinamento
│   ├── best_kmeans_metrics.json   # Métricas do melhor modelo
│   └── README.md                   # Documentação detalhada
│
├── NB/                             # Naive Bayes
│   ├── train.py                    # Script de treinamento
│   ├── best_nb_metrics.json       # Métricas do melhor modelo
│   ├── requirements.txt            # Dependências específicas
│   └── README.md                   # Documentação detalhada
│
├── RF/                             # Random Forest
│   ├── train.py                    # Script de treinamento
│   ├── best_rf_metrics.json       # Métricas do melhor modelo
│   ├── requirements.txt            # Dependências específicas
│   └── README.md                   # Documentação detalhada
│
└── SVM/                            # Support Vector Machine
    ├── train.py                    # Script de treinamento
    ├── best_svm_metrics.json      # Métricas do melhor modelo
    ├── requirements.txt            # Dependências específicas
    └── README.md                   # Documentação detalhada
```

---

## 🤖 Modelos Implementados

### 1. **FNN - Feedforward Neural Network**
- **Tipo**: Deep Learning (PyTorch)
- **Arquitetura**:
  - Camada de Entrada: 22.278 neurônios
  - Camadas Ocultas: 128 → 64 neurônios com ReLU
  - Batch Normalization e Dropout para regularização
  - Camada de Saída: 2 neurônios (softmax)
- **Hiperparâmetros Otimizados**:
  - Hidden Dim 1: [64, 128]
  - Hidden Dim 2: [32, 64]
  - Dropout Rate: [0.3, 0.5]
  - Learning Rate: [0.001, 0.0005]
- **Local**: [FNN/README.md](FNN/README.md)

### 2. **KNN - K-Nearest Neighbors**
- **Tipo**: Algoritmo baseado em distância
- **Pipeline**:
  - StandardScaler (normalização)
  - PCA (redução dimensional para 5-20 componentes)
  - KNN com múltiplas configurações
- **Hiperparâmetros Otimizados**:
  - Componentes PCA: [5, 10, 20]
  - k (vizinhos): [3, 5, 7]
  - Pesos: [uniform, distance]
  - Métrica: [euclidean, manhattan]
- **Local**: [KNN/README.md](KNN/README.md)

### 3. **KMeans - K-Means Clustering**
- **Tipo**: Algoritmo não supervisionado
- **Aplicação**: Clusterização seguida de classificação
- **Hiperparâmetros Otimizados**:
  - Número de clusters: [2, 3, 4, 5]
  - Inicialização: k-means++
- **Local**: [KM/README.md](KM/README.md)

### 4. **Naive Bayes - Gaussian Naive Bayes**
- **Tipo**: Modelo probabilístico
- **Características**:
  - Simples e rápido
  - Baseado no Teorema de Bayes
  - Assume independência entre características
- **Local**: [NB/README.md](NB/README.md)

### 5. **Random Forest**
- **Tipo**: Ensemble de árvores de decisão
- **Hiperparâmetros Otimizados**:
  - n_estimators: [100, 200]
  - max_depth: [10, 20]
  - min_samples_split: [2, 5]
  - min_samples_leaf: [1, 2]
  - max_features: [sqrt, log2]
- **Vantagens**: Reduz overfitting, rápido, bom desempenho
- **Local**: [RF/README.md](RF/README.md)

### 6. **SVM - Support Vector Machine**
- **Tipo**: Modelo discriminativo
- **Técnicas Aplicadas**:
  - SMOTE para balanceamento
  - Kernel RBF
  - Grid Search para otimização
- **Hiperparâmetros Otimizados**:
  - C: [0.01, 0.1, 1, 10, 100]
  - gamma: [0.001, 0.01, 0.1, 1, scale]
  - Kernel: RBF
- **Local**: [SVM/README.md](SVM/README.md)

---

## ⚙️ Instalação

### Pré-requisitos
- Python 3.8+
- pip ou conda

### Passo 1: Clonar ou Baixar o Repositório
```bash
git clone https://github.com/seu-usuario/GeneCancerPredictor.git
cd GeneCancerPredictor
```

### Passo 2: Criar Ambiente Virtual (Recomendado)
```bash
# Com venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Ou com conda
conda create -n cancer-predictor python=3.9
conda activate cancer-predictor
```

### Passo 3: Instalar Dependências Principais
```bash
pip install -r requirements.txt
```

**Conteúdo do requirements.txt:**
```
streamlit         # Interface web interativa
pandas           # Manipulação de dados
matplotlib       # Visualizações
seaborn          # Visualizações estatísticas
scikit-learn     # Machine Learning
```

### Passo 4: Instalar Dependências por Modelo (Opcional)

Se deseja treinar modelos específicos:

**Para FNN (Deep Learning):**
```bash
cd FNN
pip install -r requirements.txt
# Inclui: torch, pytorch
cd ..
```

**Para SVM:**
```bash
cd SVM
pip install -r requirements.txt
# Inclui: imbalanced-learn (SMOTE)
cd ..
```

**Para Random Forest:**
```bash
cd RF
pip install -r requirements.txt
cd ..
```

---

## 🚀 Como Usar

### Opção 1: Executar a Interface Streamlit (Recomendado)

```bash
streamlit run app.py
```

A interface será aberta em `http://localhost:8501`

**Recursos da Interface:**
- 📊 Visualização de métricas de todos os modelos
- 📈 Gráficos comparativos (bar, area charts)
- 🔍 Visualização PCA dos dados
- 📉 Análise de variância gênica
- 🎯 Confusão matrices por modelo
- 📋 Distribuição de classes

### Opção 2: Treinar Modelos Individuais

#### Treinar FNN
```bash
cd FNN
python train.py
cd ..
```

#### Treinar Random Forest
```bash
cd RF
python train.py
cd ..
```

#### Treinar SVM
```bash
cd SVM
python train.py
cd ..
```

#### Treinar KNN
```bash
cd KNN
python train.py
cd ..
```

#### Treinar Naive Bayes
```bash
cd NB
python train.py
cd ..
```

#### Treinar K-Means
```bash
cd KM
python train.py
cd ..
```

### Opção 3: Usar em Código Python

```python
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Carregar dados
dataset = pd.read_csv('Liver_GSE14520_U133A.csv')
X = dataset.drop(['samples', 'type'], axis=1)
y = dataset['type'].map({'HCC': 1, 'normal': 0})

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Treinar modelo
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_scaled, y)

# Fazer predições
predictions = model.predict(X_scaled)
print(f"Acurácia: {(predictions == y).mean():.3f}")
```

---

## 📊 Resultados e Comparação

### Resumo de Desempenho dos Modelos

| Modelo | Acurácia | Precisão | Recall | F1-Score | MCC |
|--------|----------|----------|--------|----------|-----|
| **FNN** | 95.8% | 97.2% | 94.5% | 95.8% | 0.916 |
| **KNN** | 95.5% | 97.7% | 93.4% | 95.5% | 0.911 |
| **Random Forest** | **96.4%** | 96.7% | **96.1%** | 96.4% | **0.927** |
| **SVM** | 95.7% | 97.8% | 93.9% | 95.7% | 0.914 |
| **Naive Bayes** | 92.1% | 94.2% | 89.6% | 91.8% | 0.842 |
| **K-Means** | 88.2% | 91.5% | 85.0% | 88.2% | 0.765 |

### Análise por Métrica

**🏆 Melhor Desempenho Geral: Random Forest**
- Melhor acurácia e recall
- MCC mais alto (0.927)
- Melhor balanço entre precisão e recall

**📌 Insights:**
1. **Supervised vs Unsupervised**: Modelos supervisionados (RF, FNN, KNN) superam K-Means
2. **Deep Learning vs Classical ML**: FNN competitivo com modelos clássicos
3. **Recall Alto**: Importante para diagnóstico médico (reduz falsos negativos)
4. **MCC**: Melhor métrica para dados desbalanceados

---

## 📈 Visualizações

A interface Streamlit oferece várias visualizações:

### 1. **Gráfico de Comparação por Métrica**
- Compara Acurácia, Precisão, Recall, F1-Score
- Formato: Bar chart
- Identifica rapidamente o melhor modelo

### 2. **Gráfico de Recall Específico**
- Foco na métrica de recall (menor taxa de falsos negativos)
- Crítico em diagnósticos médicos

### 3. **Gráfico de Área (Area Chart)**
- Visualiza tendências entre modelos
- Mostra todas as métricas simultaneamente

### 4. **Visualização PCA**
- Reduz 22.278 dimensões para 2D
- Mostra separabilidade entre classes HCC e Normal
- Coloring por classe

### 5. **Distribuição de Classes**
- Bar chart com contagem de amostras
- Mostra desbalanceamento dos dados

### 6. **Top 10 Genes com Maior Variância**
- Identifica genes mais importantes
- Contribuem mais para diferenciação

### 7. **Pairplot de Top 5 Genes**
- Análise pairwise entre genes importantes
- Colorido por classe

### 8. **Confusion Matrix**
- Matriz por modelo
- Visualiza True Positives, False Positives, etc.

---

## 🔧 Requisitos e Dependências

### Dependências Principais (requirements.txt)
```
streamlit>=1.0.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
numpy>=1.20.0
```

### Dependências Específicas por Modelo

**FNN/requirements.txt:**
```
torch>=1.10.0
pytorch>=1.10.0
```

**SVM/requirements.txt:**
```
imbalanced-learn>=0.8.0
```

**RF/requirements.txt e NB/requirements.txt:**
```
scikit-learn>=1.0.0
```

### Versões Testadas
- Python 3.8, 3.9, 3.10
- scikit-learn 1.0.0+
- PyTorch 1.10.0+
- Streamlit 1.0.0+

---

## 📝 Uso Detalhado da Interface

### Iniciar a Aplicação
```bash
streamlit run app.py
```

### Primeira Execução
1. O sistema baixará automaticamente `Liver_GSE14520_U133A.csv`
2. Pode levar alguns minutos na primeira vez
3. Mensagem de sucesso aparecerá quando completo

### Navegação
1. **Menu Superior**: Selecione diferentes visualizações
2. **Sidebar**: Algumas opções de configuração
3. **Charts Interativos**: Hover para ver valores exatos
4. **Refresh**: Atualizar página para recarregar dados

---

## 🔍 Interpretação de Métricas

### Acurácia
$$\text{Acurácia} = \frac{TP + TN}{TP + TN + FP + FN}$$
- Proporção de predições corretas
- Menos útil em dados desbalanceados

### Precisão
$$\text{Precisão} = \frac{TP}{TP + FP}$$
- De todos os positivos previstos, quantos eram corretos?
- Importante quando falsos positivos são custosos

### Recall (Sensibilidade)
$$\text{Recall} = \frac{TP}{TP + FN}$$
- De todos os casos positivos reais, quantos foram detectados?
- **CRÍTICO em diagnósticos** - reduz diagnósticos perdidos

### F1-Score
$$F1 = 2 \times \frac{\text{Precisão} \times \text{Recall}}{\text{Precisão} + \text{Recall}}$$
- Balanço entre Precisão e Recall
- Melhor para dados desbalanceados

### MCC (Matthews Correlation Coefficient)
$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$
- Melhor métrica para classificação binária desbalanceada
- Varia de -1 a 1 (1 = perfeito)

---

## 🎓 Aplicações Médicas

Este projeto pode ser usado para:

1. **Pesquisa Genômica**: Identificar padrões de expressão gênica
2. **Diagnóstico Auxiliar**: Complementar diagnósticos clínicos
3. **Predição de Risco**: Identificar pacientes de alto risco
4. **Estudos Comparativos**: Validar eficácia de diferentes abordagens
5. **Educação**: Ensinar ML em contexto biomédico

### ⚠️ Avisos Importantes
- **NÃO é substituição para diagnóstico médico**
- **Deve ser validado com dados clínicos reais**
- **Sempre consulte profissionais de saúde**
- Resultados dependem da qualidade dos dados de entrada

---

## 🐛 Solução de Problemas

### Problema: Dataset não baixa
```bash
# Verifique conexão com internet
# Ou baixe manualmente de:
# https://sbcb.inf.ufrgs.br/data/cumida/Genes/Liver/GSE14520_U133A/Liver_GSE14520_U133A.csv
```

### Problema: PyTorch não instala no Windows
```bash
# Tente instalar com conda em vez de pip
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Problema: Streamlit port já está em uso
```bash
streamlit run app.py --server.port 8502
```

### Problema: Memória insuficiente ao treinar
- Modelos usam todo o dataset (357 amostras x 22.278 genes)
- Se tiver erro de memória, reduza batch size nos scripts de treino

---

## 📚 Referências

### Datasets
- [CuMiDa Database](https://sbcb.inf.ufrgs.br/cumida)
- [Gene Expression Omnibus (GEO)](https://www.ncbi.nlm.nih.gov/geo/)

### Bibliotecas
- [scikit-learn Documentation](https://scikit-learn.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Papers Relacionados
- Machine learning for cancer prediction
- Gene expression analysis for HCC diagnosis
- Comparative studies of ML algorithms in medical diagnosis

---

## 🤝 Contribuições

Contribuições são bem-vindas! Como contribuir:

1. **Fork** o repositório
2. **Crie uma branch** para sua feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** suas mudanças (`git commit -m 'Add AmazingFeature'`)
4. **Push** para a branch (`git push origin feature/AmazingFeature`)
5. **Abra um Pull Request**

### Ideias para Contribuições
- [ ] Adicionar novos modelos (XGBoost, LightGBM)
- [ ] Melhorar visualizações
- [ ] Otimizar performance
- [ ] Adicionar interpretabilidade (SHAP, LIME)
- [ ] Validação cross-database
- [ ] Deploy em plataforma web
- [ ] Documentação em outros idiomas
- [ ] Testes automatizados

---

## 📄 Licença

Este projeto está licenciado sob a [LICENSE](LICENSE) - veja o arquivo para detalhes.

---

## 👨‍💻 Autor

**Desenvolvido por**: Neo Kavinsky

---

## 📞 Contato e Suporte

- 📧 Email: [seu-email]
- 💻 GitHub: [seu-github]
- 🐛 Issues: [link para issues]

---

## 📈 Estatísticas do Projeto

- **Modelos Implementados**: 6
- **Métricas de Avaliação**: 8+
- **Genes Analisados**: 22.278
- **Amostras**: 357
- **Acurácia Máxima**: 96.4% (Random Forest)
- **Recall Máximo**: 96.1% (Random Forest)

---

## 🎯 Roadmap Futuro

### v2.0
- [ ] Interface com upload de arquivos
- [ ] Predições em tempo real
- [ ] Análise de importância de features
- [ ] Integração com API médica

### v3.0
- [ ] Suporte para múltiplos tipos de câncer
- [ ] Análise de survival
- [ ] Integração com banco de dados
- [ ] API REST

### v4.0
- [ ] Deploy em cloud (AWS, GCP, Azure)
- [ ] Aplicativo mobile
- [ ] Integração com DICOM images
- [ ] Real-time model updates

---

**Última Atualização**: January 13, 2026

**Status do Projeto**: ✅ Ativo e em desenvolvimento

---

<div align="center">

### ⭐ Se este projeto foi útil, considere dar uma estrela! ⭐

Made with ❤️ for medical AI research

</div>