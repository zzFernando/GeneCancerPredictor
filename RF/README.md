### **Random Forest Model Performance Report**

This document outlines the methodology, metrics, and results obtained from training and evaluating a **Random Forest Classifier** for the classification task using genomic data from the **GeneCancerPredictor** project.

---

## **1. Methodology**

### **Dataset Overview**
- **Source:** [CuMiDa - Curated Microarray Database](https://sbcb.inf.ufrgs.br/cumida/), Institute of Informatics, Federal University of Rio Grande do Sul (UFRGS).
- **Dataset Used:** GSE14520_U133A
  - **Cancer Type:** Liver
  - **Platform:** GPL571
  - **Samples:** 357
  - **Genes:** 22,278
  - **Classes:** 2 (`HCC` and `Normal`).

### **Data Processing**
1. **Preprocessing:**
   - Features were standardized using `StandardScaler` to ensure equal contribution of all features during model training.
   - Target labels were binarized: `HCC = 1`, `Normal = 0`.

2. **Train-Test Split:**
   - 80% of the data was used for training, and 20% was used for testing.

3. **Model Selection:**
   - Random Forest Classifier with the following parameters:
     - `n_estimators=100`: Number of trees.
     - `max_depth=10`: Maximum depth of trees to prevent overfitting.
     - `min_samples_split=5`: Minimum samples required to split an internal node.
     - `min_samples_leaf=2`: Minimum samples required at a leaf node.
     - `max_features='sqrt'`: Features considered at each split.
     - `class_weight='balanced'`: Handles class imbalance by adjusting weights inversely proportional to class frequencies.

4. **Validation Methodology:**
   - **Stratified 10-Fold Cross-Validation**: Ensures class proportions are maintained across folds to evaluate generalization performance.

---

## **2. Metrics Explained**

### **Model Metrics**
1. **Accuracy:** Proportion of correctly classified samples over the total.
2. **Precision:** Proportion of true positive predictions out of all positive predictions.
3. **Recall (Sensitivity):** Proportion of actual positive samples correctly identified.
4. **F1-Score:** Harmonic mean of Precision and Recall.
5. **MCC (Matthews Correlation Coefficient):** Evaluates the quality of binary classifications, ranging from -1 (complete disagreement) to +1 (perfect classification).
6. **Kappa Statistic:** Measures agreement beyond chance. Values closer to 1 indicate better agreement.
7. **ROC AUC (Receiver Operating Characteristic - Area Under Curve):** Measures the ability of the model to distinguish between classes. A value close to 1.0 indicates excellent performance.
8. **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual values.
9. **Root Mean Squared Error (RMSE):** Square root of the average squared difference between predicted and actual values.

---

## **3. Results**

### **Training Set Performance**
- **Accuracy:** 0.9930
- **Precision:** 0.9863
- **Recall:** 1.0000
- **F1-Score:** 0.9931
- **MCC:** 0.9861
- **Mean Absolute Error:** 0.0070
- **Root Mean Squared Error:** 0.0838

#### **Confusion Matrix (Training Set):**
| Predicted/Classified as | HCC (1) | Normal (0) |
|--------------------------|---------|------------|
| **HCC (1)**              | 144     | 0          |
| **Normal (0)**           | 2       | 139        |

---

### **Stratified 10-Fold Cross-Validation**
- **Accuracy:** 0.9636
- **Precision:** 0.9719
- **Recall:** 0.9558
- **F1-Score:** 0.9638
- **MCC:** 0.9273
- **Kappa Statistic:** 0.9272
- **ROC AUC:** 0.9637

#### **Confusion Matrix (Cross-Validation):**
| Predicted/Classified as | HCC (1) | Normal (0) |
|--------------------------|---------|------------|
| **HCC (1)**              | 173     | 8          |
| **Normal (0)**           | 5       | 170        |

---

### **Class-Specific Performance (Cross-Validation)**
| Metric       | Normal (0) | HCC (1) |
|--------------|------------|---------|
| **Precision**| 0.96       | 0.97    |
| **Recall**   | 0.97       | 0.96    |
| **F1-Score** | 0.96       | 0.96    |
| **ROC AUC**  | 0.96       | 0.96    |

---

## **4. Observations**

1. **Training vs. Validation:**
   - The model performed exceptionally well on the training set, achieving near-perfect scores across all metrics.
   - Slight drops in performance during cross-validation indicate some degree of overfitting.

2. **Class-Specific Insights:**
   - Precision and Recall are well-balanced across both classes (HCC and Normal), suggesting the model handles class imbalance effectively.

3. **Differences in Sample Count:**
   - The **training confusion matrix** includes all 357 samples since it evaluates the model on the entire dataset.
   - The **validation confusion matrix** reflects predictions from 10-fold cross-validation, where each sample is evaluated in one of the folds.

4. **Model Robustness:**
   - High MCC (0.9273) and Kappa Statistic (0.9272) during cross-validation highlight the robustness of the model.

---

## **5. Conclusion**
- The **Random Forest Classifier** demonstrates high accuracy and robustness in classifying genomic data, achieving a Recall of 0.9558 during cross-validation.
- Despite slight overfitting in the training set, the model generalizes well, as evidenced by its cross-validation metrics.
- Future work may include feature selection and hyperparameter tuning to further reduce overfitting and enhance computational efficiency.