# Naive Bayes Classifier

## Overview
This project applies a **Naive Bayes Classifier** to predict liver cancer (Hepatocellular Carcinoma - HCC) using genomic data. The genomic dataset is sourced from the CuMiDa database and preprocessed for efficient machine learning workflows. The Naive Bayes model was optimized using grid search and evaluated using 10-fold cross-validation.

### Dataset
- **Source**: [CuMiDa Database](https://sbcb.inf.ufrgs.br/cumida)
- **Dataset ID**: GSE14520_U133A
- **Samples**: 357
- **Genes**: 22,278
- **Classes**: Binary (HCC vs. Normal)

The dataset was standardized using `StandardScaler` to normalize the gene expression values.

## Methodology
### Algorithm Setup
The Naive Bayes classifier was implemented using the following pipeline:
1. **StandardScaler**: Standardized the data to eliminate scale bias.
2. **Gaussian Naive Bayes**: Applied the GaussianNB classifier.

### Grid Search Parameters
The model was optimized with the following hyperparameters:
- **Var Smoothing**: [1e-9, 1e-8, 1e-7, 1e-6]

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
- **Accuracy**: 0.952
- **Precision**: 0.951
- **Recall**: 0.956
- **F1-Score**: 0.953
- **MCC**: 0.905
- **Kappa**: 0.905

### Confusion Matrix
```
[[167, 9],
 [  8, 173]]
```

### Best Parameters
- **Var Smoothing**: 1e-9

## Running the Model
### Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/zzFernando/GeneCancerPredictor.git
cd GeneCancerPredictor/NB
pip install -r requirements.txt
```

### Training and Evaluation
Run the training script:
```bash
python train.py
```
The script performs grid search with 10-fold cross-validation and saves the best metrics to `best_nb_metrics.json`.

## Possible Improvements
1. **Feature Selection**: Explore advanced feature selection techniques to enhance model interpretability.
2. **Kernel Density Estimation**: Extend Naive Bayes with kernel density estimation for better handling of non-Gaussian data.
3. **Data Augmentation**: Generate synthetic data to address class imbalances and improve generalization.
4. **Hybrid Models**: Combine Naive Bayes with other classifiers for ensemble methods.

---
For more information, refer to the main repository: [GeneCancerPredictor](https://github.com/zzFernando/GeneCancerPredictor)
