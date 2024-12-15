import streamlit as st
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
import urllib.request
from sklearn.decomposition import PCA

# Define dataset URL e nome do arquivo
DATASET_URL = "https://sbcb.inf.ufrgs.br/data/cumida/Genes/Liver/GSE14520_U133A/Liver_GSE14520_U133A.csv"
DATASET_FILE = "Liver_GSE14520_U133A.csv"

# Função para verificar e baixar o dataset
def check_and_download_dataset():
    if not os.path.exists(DATASET_FILE):
        st.warning("Dataset não encontrado. Iniciando o download...")
        try:
            urllib.request.urlretrieve(DATASET_URL, DATASET_FILE)
            st.success("Dataset baixado com sucesso!")
        except Exception as e:
            st.error(f"Erro ao baixar o dataset: {e}")
            raise

# Load JSON metrics
def load_metrics(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Load all model metrics
def load_all_metrics():
    models = ["FNN", "KMeans", "KNN", "Naive Bayes", "Random Forest", "SVM"]
    metrics_files = [
        "best_fnn_metrics.json",
        "best_kmeans_metrics.json",
        "best_knn_metrics.json",
        "best_nb_metrics.json",
        "best_rf_metrics.json",
        "best_svm_metrics.json",
    ]
    metrics = {}
    for model, file in zip(models, metrics_files):
        if os.path.exists(file):
            metrics[model] = load_metrics(file)
    return metrics

# Plot confusion matrix
def plot_confusion_matrix(conf_matrix, model_name):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"Confusion Matrix for {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig

def plot_recall_line(metrics):
    # Converte os dados em um DataFrame e filtra apenas o Recall
    df_metrics = pd.DataFrame(metrics).set_index("Model")
    recall_values = df_metrics["Recall"]

    # Define os limites do eixo Y com base no Recall (para evitar muito espaço em branco)
    y_min = recall_values.min() - 0.01  # Margem inferior
    y_max = recall_values.max() + 0.01  # Margem superior

    # Criação do gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(recall_values.index, recall_values, marker="o", linestyle="-", color="#FF5733", linewidth=2, label="Recall")

    # Configurações do gráfico
    ax.set_title("Recall Comparison Across Models", fontsize=15)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Recall Value", fontsize=12)
    ax.set_ylim(y_min, y_max)  # Ajusta o range do eixo Y dinamicamente
    ax.grid(True, linestyle="--", alpha=0.5)  # Adiciona grid para facilitar leitura
    ax.legend(loc="lower right")

    # Rótulos rotacionados para o eixo X
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig

# Plot performance comparison with bar chart
def plot_model_comparison_bar(metrics):
    df_metrics = pd.DataFrame(metrics).set_index("Model")
    y_min, y_max = df_metrics.min().min(), df_metrics.max().max()  # Define os limites com base nas métricas
    fig, ax = plt.subplots(figsize=(10, 6))
    df_metrics.plot(kind="bar", ax=ax)
    ax.set_title("Model Performance Bar Chart")
    ax.set_xlabel("Model")
    ax.set_ylabel("Metric Value")
    ax.set_ylim(y_min - 0.05, y_max + 0.05)  # Adiciona margem ao redor do range
    plt.xticks(rotation=45)
    return fig

def plot_model_comparison_area(metrics):
    df_metrics = pd.DataFrame(metrics).set_index("Model")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Loop para plotar cada métrica com área separada
    for metric in df_metrics.columns:
        ax.fill_between(df_metrics.index, df_metrics[metric], label=metric, alpha=0.5)
        ax.plot(df_metrics.index, df_metrics[metric], marker="o", linestyle="-", label=f"{metric} (Line)")
    
    ax.set_title("Model Performance Area Chart")
    ax.set_xlabel("Model")
    ax.set_ylabel("Metric Value")
    
    # Ajuste dinâmico do limite do eixo y
    y_min, y_max = df_metrics.min().min(), df_metrics.max().max()
    ax.set_ylim(y_min - 0.05, y_max + 0.05)
    
    plt.xticks(rotation=45)
    plt.legend(title="Metrics", loc="upper left", bbox_to_anchor=(1, 1))  # Colocar a legenda fora do gráfico
    plt.tight_layout()
    return fig

# Gene expression distribution using PCA
def plot_pca_distribution():
    data = pd.read_csv("Liver_GSE14520_U133A.csv")
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data.iloc[:, 2:])
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=pd.factorize(data['type'])[0], cmap="coolwarm", alpha=0.7)
    ax.set_title("PCA Visualization of Gene Expression")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    legend = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend)
    return fig

# Samples per class
def plot_samples_per_class():
    data = pd.read_csv("Liver_GSE14520_U133A.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    data["type"].value_counts().plot(kind="bar", ax=ax, color=["blue", "orange"])
    ax.set_title("Samples per Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    return fig

# Distribution of top genes by variance
def plot_top_genes_variance():
    data = pd.read_csv("Liver_GSE14520_U133A.csv")
    variances = data.iloc[:, 2:].var().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=variances.values, y=variances.index, palette="viridis", ax=ax)
    ax.set_title("Top 10 Genes with Highest Variance")
    ax.set_xlabel("Variance")
    ax.set_ylabel("Gene Index")
    return fig

# Pairplot for top 10 genes by variance
def plot_top_genes_pairplot():
    data = pd.read_csv("Liver_GSE14520_U133A.csv")
    top_genes = data.iloc[:, 2:].var().sort_values(ascending=False).head(5).index
    pairplot_data = data[top_genes]
    pairplot_data['type'] = data['type']
    fig = sns.pairplot(pairplot_data, hue="type", palette="coolwarm", diag_kind="kde")
    fig.fig.suptitle("Pairplot of Top 5 Genes by Variance", y=1.02)
    return fig

# Main Streamlit app
st.title("Model Comparison and Visualizations")

# Check and download dataset if necessary
check_and_download_dataset()

# Load metrics
st.sidebar.title("Select an Option")
metrics = load_all_metrics()

# Sidebar for navigation
option = st.sidebar.selectbox("Choose a task", ["Model Comparison", "Dataset Visualizations"])

if option == "Model Comparison":
    st.header("Model Performance Comparison")
    if metrics: 
        results = []
        for model, data in metrics.items():
            results.append({
                "Model": model,
                "Accuracy": data["Accuracy"],
                "Precision": data["Precision"],
                "Recall": data["Recall"],
                "F1-Score": data["F1-Score"],
                "MCC": data["MCC"]
            })
        df_metrics = pd.DataFrame(results)

        # Ordena a tabela pela métrica "Recall" em ordem decrescente
        df_metrics = df_metrics.sort_values(by="Recall", ascending=False)

        st.table(df_metrics)

        # Add comparison charts in columns
        st.subheader("Model Performance Charts")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Line Chart")
            fig_line = plot_recall_line(results)
            st.pyplot(fig_line)
        with col2:
            st.subheader("Bar Chart")
            fig_bar = plot_model_comparison_bar(results)
            st.pyplot(fig_bar)
        with col3:
            st.subheader("Area Chart")
            fig_area = plot_model_comparison_area(results)
            st.pyplot(fig_area)

        # Select model for detailed analysis (separate from columns)
        st.subheader("Detailed Model Analysis")
        selected_model = st.selectbox("Select a model for detailed analysis", metrics.keys())
        if selected_model:
            model_data = metrics[selected_model]
            st.subheader(f"{selected_model} Details")
            st.json(model_data["params"])
            st.write(f"Confusion Matrix for {selected_model}")
            fig = plot_confusion_matrix(model_data["Confusion Matrix"], selected_model)
            st.pyplot(fig)
    else:
        st.warning("No model metrics files found!")

elif option == "Dataset Visualizations":
    st.header("Dataset Visualizations")

    # PCA distribution
    st.subheader("PCA Visualization of Gene Expression")
    fig_pca_distribution = plot_pca_distribution()
    st.pyplot(fig_pca_distribution)
    st.caption("This scatter plot shows the PCA-reduced dimensions of the gene expression data, colored by class.")

    # Samples per class
    st.subheader("Samples per Class")
    fig_samples_class = plot_samples_per_class()
    st.pyplot(fig_samples_class)
    st.caption("This bar chart represents the number of samples in each class (HCC vs. Normal).")

    # Top genes by variance
    st.subheader("Top 10 Genes with Highest Variance")
    fig_top_genes_variance = plot_top_genes_variance()
    st.pyplot(fig_top_genes_variance)
    st.caption("This bar chart shows the top 10 genes with the highest variance in their expression levels.")

    # Pairplot of top 5 genes
    st.subheader("Pairplot of Top 5 Genes by Variance")
    fig_top_genes_pairplot = plot_top_genes_pairplot()
    st.pyplot(fig_top_genes_pairplot.fig)
    st.caption("This pairplot shows relationships between the top 10 genes with the highest variance, colored by class.")
