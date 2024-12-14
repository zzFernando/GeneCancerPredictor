# K-Means 

## Overview
This module employs **K-Means Clustering** to classify liver cancer (Hepatocellular Carcinoma - HCC) using gene expression data. The genomic dataset, sourced from the **CuMiDa** database, undergoes grid search optimization to achieve the best clustering performance.

---

## Dataset Details
- **Source**: [CuMiDa Database](https://sbcb.inf.ufrgs.br/cumida)
- **Dataset ID**: GSE14520_U133A
- **Platform**: GPL571
- **Samples**: 357
- **Genes**: 22,278
- **Classes**: Binary (HCC vs. Normal)

**Note**: The CuMiDa database ensures rigorous preprocessing, including normalization and background correction, making it ideal for machine learning tasks.

---

## Methodology

### Clustering Approach
- **Algorithm**: K-Means with two clusters (HCC and Normal), matched to actual classes via majority voting.
- **Preprocessing**: StandardScaler was used to normalize gene expression values.
- **Optimization**: Grid search over hyperparameters, evaluated using 10-fold cross-validation.

### Grid Search Parameters
| Parameter           | Values            |
|---------------------|-------------------|
| Initialization      | `['k-means++', 'random']` |
| Maximum Iterations  | `[300, 500]`      |
| Number of Initializations | `[10, 20]`   |

---

## Results

### Best Model Performance
- **Accuracy**: 0.905
- **Precision**: 0.957
- **Recall**: 0.851
- **F1-Score**: 0.901
- **MCC**: 0.815
- **Cohen's Kappa**: 0.810
- **Mean Absolute Error**: 0.095
- **Root Mean Squared Error**: 0.309

### Confusion Matrix
```
[[169, 7],
 [ 27, 154]]
```

**Best Hyperparameters**:
```json
{
    "init": "k-means++",
    "max_iter": 300,
    "n_init": 10
}
```

---

## Running the Model

### Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/zzFernando/GeneCancerPredictor.git
cd GeneCancerPredictor/KMeans
pip install -r requirements.txt
```

### Training and Evaluation
Run the training script:
```bash
python train.py
```
- **Functionality**: Executes grid search for K-Means clustering and evaluates the model using 10-fold cross-validation.
- **Output**: The best metrics are saved in `best_kmeans_metrics.json`.

---

## Possible Improvements
1. **Dimensionality Reduction**: Employ PCA or t-SNE to improve separability of clusters.
2. **Enhanced Clustering Methods**: Experiment with hierarchical or DBSCAN clustering for comparative performance.
3. **Hybrid Approaches**: Combine K-Means with supervised classifiers for semi-supervised learning.
4. **Cluster Evaluation Metrics**: Include silhouette scores or Davies-Bouldin index to validate cluster cohesion and separation.

---

## References
This module is part of the larger [GeneCancerPredictor](https://github.com/zzFernando/GeneCancerPredictor) project, exploring machine learning methods for cancer genomics.

For dataset details, visit the [CuMiDa Database](https://sbcb.inf.ufrgs.br/cumida).

### Citation
If you use the CuMiDa database, please cite:
```
@article{cumida:2019,
  title        = {CuMiDa: An Extensively Curated Microarray Database for Benchmarking and Testing of Machine Learning Approaches in Cancer Research},
  author       = {Feltes, B.C. and Chandelier, E. B. and Grisci, B. I. and Dorn, M.},
  year         = 2019,
  journal      = {Journal of Computational Biology},
  volume       = 26,
  number       = 4,
  pages        = {376--386},
  doi          = {10.1089/cmb.2018.0238}
}
```

---