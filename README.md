# **GeneCancerPredictor** 🌟

**GeneCancerPredictor** is an open-source project leveraging modern machine learning and statistical methods to predict liver cancer (Hepatocellular Carcinoma - HCC) using genomic data. This repository focuses on reproducibility, efficient workflows, and practical insights for cancer research applications.

---

## 📚 **Overview**

This project uses data from the **[CuMiDa Database](https://sbcb.inf.ufrgs.br/cumida)**, a rigorously curated resource of cancer-related microarray datasets. The specific dataset analyzed in this study is:

- **Type**: Liver Cancer (HCC vs. Normal)  
- **GSE ID**: `14520_U133A`  
- **Platform**: `GPL571`  
- **Samples**: 357  
- **Genes**: 22,278  
- **Classes**: Binary (HCC, Normal)  

CuMiDa ensures data quality through uniform preprocessing, background correction, and normalization, making it an ideal benchmark for machine learning models.

---

## ⚙️ **Methodologies**

### **1. Data Preprocessing**
- **Standardization**:  
  Data was normalized using `StandardScaler` to remove scale biases across genes.

### **2. Machine Learning Models**
The following algorithms were implemented, with hyperparameter optimization via **Grid Search** and **10-Fold Cross-Validation**:

1. **Feedforward Neural Network (FNN)**  
   - Architecture includes dense layers with dropout to prevent overfitting.  
   - Trained using `CrossEntropyLoss` and `AdamW`.

2. **Random Forest (RF)**  
   - Optimized configuration with adjusted tree counts, maximum depth, and split criteria.

3. **K-Nearest Neighbors (KNN)**  
   - Tested with different distance metrics (`euclidean`, `manhattan`).

4. **Naive Bayes (NB)**  
   - Fine-tuned `var_smoothing` for better performance.

5. **K-Means Clustering**  
   - Unsupervised clustering for exploratory analysis.

6. **Support Vector Machines (SVM)**  
   - Kernel `rbf` with optimized `C` and `gamma` parameters.

---

## 🌟 **Results**

### **Top-Performing Models (Ordered by Recall)**

| **Model**               | **Recall** | **Accuracy** | **Precision** | **F1-Score** | **MCC**  |  
|--------------------------|------------|--------------|---------------|--------------|----------|  
| **Random Forest (RF)**   | **0.961**  | **0.964**    | **0.967**     | **0.964**    | **0.927** |  
| **Naive Bayes (NB)**      | **0.956**  | **0.952**    | **0.951**     | **0.953**    | **0.905** |  
| **Support Vector Machine (SVM)** | **0.956**  | **0.947**    | **0.940**     | **0.948**    | **0.894** |  
| **Feedforward Neural Network (FNN)** | **0.945**  | **0.958**    | **0.972**     | **0.958**    | **0.916** |  
| **K-Nearest Neighbors (KNN)** | **0.934**  | **0.955**    | **0.977**     | **0.955**    | **0.911** |  
| **K-Means Clustering**    | **0.851**  | **0.905**    | **0.957**     | **0.901**    | **0.815** |

---

## 🚀 **How to Run**

### **1. Clone the Repository**
```bash
git clone https://github.com/zzFernando/GeneCancerPredictor.git  
cd GeneCancerPredictor  
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt  
```

### **3. Run Specific Models**
For example, to train the Random Forest model:
```bash
cd RF  
python train.py  
```

### **4. Access Results**
- **Metrics**: JSON files with detailed performance metrics.

---

## 🔧 **Future Improvements**

1. **Advanced Neural Networks**  
   - Introduce models like **CNNs** or **transformers** for complex pattern analysis in genomic data.

2. **Result Interpretation**  
   - Use techniques like **SHAP** to identify the most relevant genes for classification.

3. **Data Augmentation**  
   - Explore synthetic data generation techniques to improve model generalization.

4. **Ensemble Methods**  
   - Combine models for more robust predictions.

---

## 📜 **Acknowledgments**

The data used in this study comes from the **[CuMiDa Database](https://sbcb.inf.ufrgs.br/cumida)**, as described in:

**Feltes et al.** (2019). *CuMiDa: An Extensively Curated Microarray Database for Benchmarking and Testing of Machine Learning Approaches in Cancer Research*.  
**Journal of Computational Biology**, 26(4), 376-386.  
**DOI**: [10.1089/cmb.2018.0238](https://doi.org/10.1089/cmb.2018.0238)

---

## 🤝 **Contributions**

Contributions are welcome!  
Feel free to open **issues** or submit **pull requests** to improve the project.

**Thank you for supporting cancer research!** 🌟  
