### **KNN-PCA Model Performance Report**

This document outlines the methodology, metrics, and results obtained from training and evaluating a **KNN-PCA** model for the classification task using genomic data from the **GeneCancerPredictor** project.

---

## **Dataset Description**
The dataset used in this study originates from the [CuMiDa: Curated Microarray Database](https://sbcb.inf.ufrgs.br/data/cumida/Genes/Liver/). It is a carefully curated resource containing genomic data for cancer research, optimized for machine learning applications. Specifically, the dataset `Liver_GSE14520_U133A` was used, featuring:

- **Genes:** 22,278
- **Samples:** 357
- **Classes:** 2 (HCC and Normal)

---

## **Model Description**
The **K-Means Clustering** algorithm was employed to classify the genomic data into two clusters. Since K-Means is an unsupervised learning algorithm, cluster labels were aligned with the true labels in the training set using majority voting to map clusters to class labels (`HCC` and `Normal`).

### **Key Parameters**
- **Weights:** `uniform`
- **Number of neighbors:** 3
- **Metric:** `euclidean`
- **Random state:** 42 (to ensure reproducibility)

---

## **Performance Metrics**

### **1. Training on the Complete Dataset**
On the training dataset, the model achieved the following results:

| Metric                   | Value     |
|--------------------------|-----------|
| **Accuracy**             | 0.9614    |
| **Precision**            | 0.9784    |
| **Recall**               | 0.9444    |
| **F1-Score**             | 0.9611    |
| **Matthews Correlation Coefficient (MCC)** | 0.9234    |
| **Mean Absolute Error (MAE)** | 0.0386    |
| **Root Mean Squared Error (RMSE)** | 0.1965    |

#### Confusion Matrix:
```
[[138   3]
 [  8 136]]
```

---

### **2. Stratified Cross-Validation**
Using 10-fold Stratified Cross-Validation, the model performance was evaluated across all samples:

| Metric                   | Value     |
|--------------------------|-----------|
| **Accuracy**             | 0.9552    |
| **Precision**            | 0.9769    |
| **Recall**               | 0.9337    |
| **F1-Score**             | 0.9548    |
| **Matthews Correlation Coefficient (MCC)** | 0.9113   |
| **Cohen's Kappa**        | 0.9104   |

#### Confusion Matrix:
```
[[172   4]
 [ 12 169]]
```

---

### **3. Class-Specific Performance**
The detailed performance per class (Normal and HCC) from cross-validation is shown below:

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| **Normal**  | 0.93      | 0.98   | 0.96     | 176     |
| **HCC**     | 0.98      | 0.93   | 0.95     | 181     |

- **Accuracy:** 0.96  
- **Macro Average:** Precision = 0.96, Recall = 0.96, F1-Score = 0.96  
- **Weighted Average:** Precision = 0.96, Recall = 0.96, F1-Score = 0.96  

---

## **Observations**
1. **Training Results:** The K-Means model performed relatively well on the training dataset, achieving high precision and recall values.
2. **Cross-Validation:** The performance on the cross-validation set was significantly lower, indicating potential issues with generalization due to the unsupervised nature of K-Means.
3. **Clustering Challenges:** The high-dimensional nature of genomic data and the absence of supervision during clustering likely contributed to the observed performance gaps.
4. **Class Imbalance Impact:** Despite balanced classes in the dataset, K-Means does not inherently account for label balance, impacting precision and recall for certain folds.

---

## **Future Improvements**
1. **Feature Selection:** Reducing the dimensionality of the dataset using PCA or selecting highly relevant genes could improve clustering performance.
2. **Alternative Models:** Exploring semi-supervised or supervised clustering techniques (e.g., Gaussian Mixture Models or DBSCAN with label propagation).
3. **Hyperparameter Tuning:** Further tuning initialization strategies, distance metrics, and cluster mapping processes.
4. **Dimensionality Reduction:** Employing t-SNE or UMAP for feature transformation to aid clustering in high-dimensional spaces.

---

## **Conclusion**
The K-Means clustering model provided moderate results, especially on the training dataset, but exhibited limitations during cross-validation. Future work will focus on enhancing feature engineering and exploring supervised models to address these challenges.