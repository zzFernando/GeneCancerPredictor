### **Feedforward Neural Network Performance Report**

This document outlines the methodology, metrics, and results obtained from training and evaluating a **Feedforward Neural Network (FNN)** model for the classification of genomic data from the **GeneCancerPredictor** project.

---

## **Dataset Description**
The dataset used in this study originates from the [CuMiDa: Curated Microarray Database](https://sbcb.inf.ufrgs.br/data/cumida/Genes/Liver/). It is a carefully curated resource containing genomic data optimized for machine learning applications. Specifically, the dataset `Liver_GSE14520_U133A` was utilized, featuring:

- **Genes:** 22,278  
- **Samples:** 357  
- **Classes:** 2 (HCC and Normal)

---

## **Model Description**
The **Feedforward Neural Network (FNN)** was trained and validated to classify genomic data into two classes: `HCC` (Hepatocellular Carcinoma) and `Normal`. The network consists of fully connected layers with Batch Normalization and Dropout for regularization.

### **Model Architecture**
1. **Input Layer:** 22,278 features (genes)  
2. **Hidden Layer 1:** 128 neurons, Batch Normalization, ReLU activation, Dropout (30%)  
3. **Hidden Layer 2:** 64 neurons, Batch Normalization, ReLU activation, Dropout (30%)  
4. **Output Layer:** 2 neurons, Softmax activation

### **Training Configuration**
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** AdamW (learning rate = 0.001, weight decay = 1e-4)  
- **Batch Size:** 64  
- **Epochs:** 20 for training and 10 for cross-validation  
- **Random State:** 42 (for reproducibility)

---

## **Performance Metrics**

### **1. Training on the Complete Dataset**
The model achieved the following results when trained on the entire dataset:

| Metric                   | Value   |
|--------------------------|---------|
| **Accuracy**             | 1.0000  |
| **Precision**            | 1.0000  |
| **Recall**               | 1.0000  |
| **F1-Score**             | 1.0000  |
| **Matthews Correlation Coefficient (MCC)** | 1.0000  |
| **ROC AUC**              | 1.0000  |

#### Confusion Matrix:
```
[[141   0]
 [  0 144]]
```

---

### **2. Stratified Cross-Validation**
Using 10-fold Stratified Cross-Validation, the model’s generalization performance was evaluated:

| Metric                   | Value   |
|--------------------------|---------|
| **Accuracy**             | 0.9720  |
| **Precision**            | 0.9831  |
| **Recall**               | 0.9613  |
| **F1-Score**             | 0.9721  |
| **Matthews Correlation Coefficient (MCC)** | 0.9442  |
| **ROC AUC**              | 0.9721  |

#### Confusion Matrix:
```
[[173   3]
 [  7 174]]
```

---

### **3. Detailed Classification Report**
The per-class performance metrics from cross-validation are summarized below:

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| **Normal**  | 0.96      | 0.98   | 0.97     | 176     |
| **HCC**     | 0.98      | 0.96   | 0.97     | 181     |

- **Overall Accuracy:** 0.97  
- **Macro Average:** Precision = 0.97, Recall = 0.97, F1-Score = 0.97  
- **Weighted Average:** Precision = 0.97, Recall = 0.97, F1-Score = 0.97  

---

## **Observations**
1. **Training Performance:** The model achieved perfect metrics (1.0000) on the training dataset, indicating successful optimization during training.
2. **Cross-Validation:** The model performed exceptionally well during cross-validation, achieving an accuracy of 97.2%, suggesting excellent generalization.
3. **Class Balance:** Balanced performance metrics for both `HCC` and `Normal` classes indicate no significant bias towards either class.
4. **Regularization Success:** The use of Dropout and Batch Normalization likely contributed to the high performance and robustness of the model.

---

## **Future Improvements**
1. **Dimensionality Reduction:** Apply Principal Component Analysis (PCA) or feature selection techniques to reduce the dimensionality of the dataset, potentially improving computational efficiency.
2. **Hyperparameter Optimization:** Explore grid search or Bayesian optimization to further fine-tune model parameters.
3. **Additional Regularization:** Experiment with L1/L2 regularization and alternative architectures, such as Convolutional Neural Networks (CNNs) for genomic data.
4. **Comparison with Other Models:** Benchmark the Feedforward Neural Network against other supervised methods, such as Random Forests or Gradient Boosting.

---

## **Conclusion**
The Feedforward Neural Network demonstrated outstanding performance on genomic data, achieving near-perfect results in training and cross-validation. The model’s ability to generalize well across folds highlights its potential for clinical applications in predicting Hepatocellular Carcinoma. Further research will explore advanced preprocessing and additional architectures to enhance performance.