# Random Forest Classifier

## Overview
This project applies a **Random Forest Classifier** to predict liver cancer (Hepatocellular Carcinoma - HCC) using genomic data. The genomic dataset is sourced from the CuMiDa database and preprocessed for efficient machine learning workflows. The Random Forest model was optimized using grid search and evaluated using 10-fold cross-validation.

### Dataset
- **Source**: [CuMiDa Database](https://sbcb.inf.ufrgs.br/cumida)
- **Dataset ID**: GSE14520_U133A
- **Samples**: 357
- **Genes**: 22,278
- **Classes**: Binary (HCC vs. Normal)

The dataset was standardized using `StandardScaler` to normalize the gene expression values.

## Methodology
### Algorithm Setup
The Random Forest classifier was implemented with the following steps:
1. **StandardScaler**: Standardized the data to eliminate scale bias.
2. **Random Forest**: Applied the RandomForestClassifier with various hyperparameter settings.

### Grid Search Parameters
The model was optimized with the following hyperparameters:
- **Number of Estimators**: [100, 200]
- **Maximum Depth**: [10, 20]
- **Minimum Samples Split**: [2, 5]
- **Minimum Samples Leaf**: [1, 2]
- **Maximum Features**: ['sqrt', 'log2']
- **Class Weight**: ['balanced']

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
- **Accuracy**: 0.964
- **Precision**: 0.967
- **Recall**: 0.961
- **F1-Score**: 0.964
- **MCC**: 0.927
- **Kappa**: 0.927

### Confusion Matrix
```
[[170, 6],
 [  7, 174]]
```

### Best Parameters
- **Number of Estimators**: 100
- **Maximum Depth**: 10
- **Minimum Samples Split**: 2
- **Minimum Samples Leaf**: 1
- **Maximum Features**: log2
- **Class Weight**: balanced

## Running the Model
### Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/zzFernando/GeneCancerPredictor.git
cd GeneCancerPredictor/RF
pip install -r requirements.txt
```

### Training and Evaluation
Run the training script:
```bash
python train.py
```
The script performs grid search with 10-fold cross-validation and saves the best metrics to `best_rf_metrics.json`.

## Possible Improvements
1. **Feature Importance Analysis**: Use SHAP values to gain interpretable insights into feature contributions.
2. **Hyperparameter Optimization**: Leverage advanced techniques like Bayesian optimization for more efficient tuning.
3. **Ensemble Learning**: Combine Random Forest with other models for better robustness.
4. **Dimensionality Reduction**: Experiment with alternative methods such as t-SNE or autoencoders to reduce dimensionality.

---
For more information, refer to the main repository: [GeneCancerPredictor](https://github.com/zzFernando/GeneCancerPredictor)
