# Naive Bayes for Liver Cancer Classification

## Overview

This document details the methods and rationale behind using a **Naive Bayes** classifier to distinguish between liver cancer (`HCC`) and normal tissue samples. The focus of this model is simplicity and interpretability, leveraging probabilistic principles to handle gene expression data.

---

## Methodology

### 1. **Model Selection: Naive Bayes**

#### Why Naive Bayes?
- **Simplicity**: A probabilistic approach that is computationally efficient, making it suitable for large datasets like gene expression profiles.
- **Assumption**: Assumes feature independence, which may not strictly hold but can still provide robust results in high-dimensional biological data.
- **Probabilistic Insights**: Outputs class probabilities, which are interpretable and valuable for clinical decision-making.

#### Limitations:
- Naive Bayes struggles when features are highly correlated, as the independence assumption is violated. However, in this application, its speed and simplicity outweigh this limitation for exploratory purposes.

---

### 2. **Feature Selection**

#### Why Feature Selection?
- **High Dimensionality**: Gene expression datasets often have tens of thousands of features, which can lead to overfitting and computational inefficiency.
- **Improved Performance**: Selecting the most important features focuses the model on relevant data, enhancing both accuracy and interpretability.

#### Method Used:
- **Random Forest Feature Importance**:
  - A `RandomForestClassifier` was used to rank features based on their importance.
  - Features above the mean importance threshold were retained, significantly reducing dimensionality while preserving critical information.

---

### 3. **Preprocessing**

- **Scaling**: Features were standardized using `StandardScaler` to zero mean and unit variance. This ensures numerical stability and improves the performance of Naive Bayes.
- **Label Encoding**: The target variable (`HCC` or `normal`) was encoded as binary values (`1` for `HCC`, `0` for `normal`).

---

### 4. **Hyperparameter Tuning**

#### Why Tune?
To optimize the performance of the Naive Bayes classifier by adjusting the `var_smoothing` parameter, which controls numerical stability during likelihood estimation.

#### Tuning Method:
- Grid Search over `var_smoothing` values: `[1e-9, 1e-8, 1e-7]`.
- Evaluated using stratified cross-validation with repeated splits for robust performance estimation.

---

### 5. **Evaluation**

#### Metrics Used:
1. **Accuracy**: To assess the overall correctness of the model.
2. **ROC AUC**: To evaluate the model's ability to distinguish between the two classes.
3. **Confusion Matrix**: Provides detailed insights into false positives and false negatives, crucial for understanding errors in a clinical context.
4. **Classification Report**: Summarizes precision, recall, and F1-score for each class.

#### Why These Metrics?
- In medical applications, sensitivity (recall for `HCC`) and specificity are critical to minimize misdiagnosis risks.
- The ROC AUC metric captures the trade-off between true positive rate and false positive rate, which is particularly important for imbalanced datasets.

---

## Results

- **Feature Dimensionality**: Reduced from tens of thousands to ~600 features after selection.
- **ROC AUC**: Demonstrates the model's discriminatory power.
- **Confusion Matrix**: Highlights the balance between true and false classifications, with a focus on minimizing false negatives for `HCC`.

---

## Conclusion

The Naive Bayes model, despite its simplicity, provides a robust baseline for liver cancer classification. By combining probabilistic predictions with effective feature selection and preprocessing, this approach achieves interpretable results suitable for exploratory analysis and rapid prototyping.