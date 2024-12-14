# Feedforward Neural Network (FNN)

## Overview
This project applies a **Feedforward Neural Network (FNN)** to predict liver cancer (Hepatocellular Carcinoma - HCC) using genomic data. The genomic dataset is sourced from the CuMiDa database and preprocessed for efficient machine learning workflows. The FNN model was optimized using grid search and evaluated using 10-fold cross-validation.

### Dataset
- **Source**: [CuMiDa Database](https://sbcb.inf.ufrgs.br/cumida)
- **Dataset ID**: GSE14520_U133A
- **Samples**: 357
- **Genes**: 22,278
- **Classes**: Binary (HCC vs. Normal)

The dataset was standardized using `StandardScaler` to normalize the gene expression values.

## Methodology
### Algorithm Setup
The FNN was implemented with the following structure:
1. **StandardScaler**: Standardized the data to eliminate scale bias.
2. **Feedforward Neural Network**:
   - **Input Layer**: Matching the number of genes.
   - **Hidden Layers**: Two layers with ReLU activations.
   - **Dropout**: To prevent overfitting.
   - **Batch Normalization**: To improve convergence.
   - **Output Layer**: Binary classification output.

### Grid Search Parameters
The model was optimized with the following hyperparameters:
- **Hidden Dimensions 1**: [64, 128]
- **Hidden Dimensions 2**: [32, 64]
- **Dropout Rate**: [0.3, 0.5]
- **Learning Rate**: [0.001, 0.0005]
- **Weight Decay**: [1e-4, 1e-5]

### Evaluation Metrics
The model's performance was evaluated using:
- **Accuracy**
- **Precision**
- **Recall** (primary metric for optimization)
- **F1-Score**
- **Matthews Correlation Coefficient (MCC)**
- **ROC AUC**
- **Confusion Matrix**

## Results
Best performance (based on Recall):
- **Accuracy**: 0.958
- **Precision**: 0.972
- **Recall**: 0.945
- **F1-Score**: 0.958
- **MCC**: 0.916
- **ROC AUC**: 0.958

### Confusion Matrix
```
[[171,  5],
 [ 10, 171]]
```

### Best Parameters
- **Hidden Dimensions 1**: 64
- **Hidden Dimensions 2**: 64
- **Dropout Rate**: 0.3
- **Learning Rate**: 0.001
- **Weight Decay**: 0.0001

## Running the Model
### Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/zzFernando/GeneCancerPredictor.git
cd GeneCancerPredictor/FNN
pip install -r requirements.txt
```

### Training and Evaluation
Run the training script:
```bash
python train.py
```
The script performs grid search with 10-fold cross-validation and saves the best metrics to `best_fnn_metrics.json`.

## Possible Improvements
1. **Advanced Architectures**: Explore convolutional or recurrent layers for better feature extraction.
2. **Data Augmentation**: Synthesize additional data to improve generalization.
3. **Optimization Techniques**: Use learning rate schedulers or optimizers like AdamW with adaptive weight decay.
4. **Explainability**: Apply SHAP or LIME to interpret the contribution of key features.

---
For more information, refer to the main repository: [GeneCancerPredictor](https://github.com/zzFernando/GeneCancerPredictor)
