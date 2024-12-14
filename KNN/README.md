# K-Nearest Neighbors (KNN) Classifier

## Overview
This project applies a **K-Nearest Neighbors (KNN)** classifier to predict liver cancer (Hepatocellular Carcinoma - HCC) using genomic data. The genomic dataset is sourced from the CuMiDa database and preprocessed for efficient machine learning workflows. The KNN model was tuned through grid search and evaluated using 10-fold cross-validation.

### Dataset
- **Source**: [CuMiDa Database](https://sbcb.inf.ufrgs.br/cumida)
- **Dataset ID**: GSE14520_U133A
- **Samples**: 357
- **Genes**: 22,278
- **Classes**: Binary (HCC vs. Normal)

The dataset was standardized using `StandardScaler`, and dimensionality reduction was performed using PCA.

## Methodology
### Algorithm Setup
The KNN classifier was implemented using the following pipeline:
1. **StandardScaler**: Standardized the data to eliminate scale bias.
2. **PCA**: Reduced the dimensionality of the data for computational efficiency.
3. **KNN**: Used the K-Nearest Neighbors algorithm for classification.

### Grid Search Parameters
The model was optimized with the following hyperparameters:
- **Number of PCA Components**: [5, 10, 20]
- **Number of Neighbors (k)**: [3, 5, 7]
- **Weights**: ['uniform', 'distance']
- **Distance Metric**: ['euclidean', 'manhattan']

### Evaluation Metrics
The model's performance was evaluated using:
- **Accuracy**
- **Precision**
- **Recall** (primary metric for optimization)
- **F1-Score**
- **Matthews Correlation Coefficient (MCC)**
- **Cohen's Kappa**
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **Confusion Matrix**

## Results
Best performance (based on Recall):
- **Accuracy**: 0.955
- **Precision**: 0.977
- **Recall**: 0.934
- **F1-Score**: 0.955
- **MCC**: 0.911
- **Kappa**: 0.910

### Confusion Matrix
```
[[172, 4],
 [ 12, 169]]
```

### Best Parameters
- **Metric**: Euclidean
- **PCA Components**: 5
- **Number of Neighbors (k)**: 3
- **Weights**: Uniform

## Running the Model
### Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/zzFernando/GeneCancerPredictor.git
cd GeneCancerPredictor/KNN
pip install -r requirements.txt
```

### Training and Evaluation
Run the training script:
```bash
python train.py
```
The script performs grid search with 10-fold cross-validation and saves the best metrics to `best_knn_metrics.json`.

## Possible Improvements
1. **Advanced Distance Metrics**: Incorporate alternative distance metrics like Minkowski or Mahalanobis.
2. **Feature Engineering**: Explore domain-specific feature selection techniques to improve model performance.
3. **Hybrid Models**: Combine KNN with other algorithms (e.g., ensemble methods) to enhance robustness.
4. **Dimensionality Reduction**: Experiment with alternative methods such as t-SNE or autoencoders.

---
For more information, refer to the main repository: [GeneCancerPredictor](https://github.com/zzFernando/GeneCancerPredictor)
