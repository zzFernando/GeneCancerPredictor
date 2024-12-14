# Support Vector Machine (SVM) Classifier

## Overview
This project applies a **Support Vector Machine (SVM)** classifier to predict liver cancer (Hepatocellular Carcinoma - HCC) using genomic data. The genomic dataset is sourced from the CuMiDa database and preprocessed for efficient machine learning workflows. The SVM model was optimized using grid search and evaluated using 10-fold cross-validation.

### Dataset
- **Source**: [CuMiDa Database](https://sbcb.inf.ufrgs.br/cumida)
- **Dataset ID**: GSE14520_U133A
- **Samples**: 357
- **Genes**: 22,278
- **Classes**: Binary (HCC vs. Normal)

The dataset was standardized using `StandardScaler` to normalize the gene expression values.

## Methodology
### Algorithm Setup
The SVM classifier was implemented with the following steps:
1. **StandardScaler**: Standardized the data to eliminate scale bias.
2. **SVM**: Applied the Support Vector Classifier with a radial basis function (RBF) kernel.

### Grid Search Parameters
The model was optimized with the following hyperparameters:
- **C**: [0.1, 1, 10] (Regularization parameter)
- **Gamma**: [0.01, 0.1, 1] (Kernel coefficient for RBF)

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
- **Accuracy**: 0.507
- **Precision**: 0.507
- **Recall**: 1.0
- **F1-Score**: 0.673
- **MCC**: 0.0
- **Kappa**: 0.0

### Confusion Matrix
```
[[  0, 176],
 [  0, 181]]
```

### Best Parameters
- **C**: 0.1
- **Gamma**: 0.01

## Running the Model
### Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/zzFernando/GeneCancerPredictor.git
cd GeneCancerPredictor/SVM
pip install -r requirements.txt
```

### Training and Evaluation
Run the training script:
```bash
python train.py
```
The script performs grid search with 10-fold cross-validation and saves the best metrics to `best_svm_metrics.json`.

## Possible Improvements
1. **Feature Engineering**: Enhance the dataset by identifying and including more relevant features.
2. **Kernel Exploration**: Experiment with other kernels such as polynomial or sigmoid.
3. **Class Imbalance Handling**: Use oversampling or synthetic data generation methods to address class imbalance.
4. **Hyperparameter Optimization**: Leverage advanced techniques like Bayesian optimization or genetic algorithms for more efficient tuning.

---
For more information, refer to the main repository: [GeneCancerPredictor](https://github.com/zzFernando/GeneCancerPredictor)
