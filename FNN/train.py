import os
import json
import urllib.request
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from joblib import Parallel, delayed

def set_deterministic():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_deterministic()

dataset_url = "https://sbcb.inf.ufrgs.br/data/cumida/Genes/Liver/GSE14520_U133A/Liver_GSE14520_U133A.csv"
file_path = 'Liver_GSE14520_U133A.csv'
if not os.path.exists(file_path):
    urllib.request.urlretrieve(dataset_url, file_path)
dataset = pd.read_csv(file_path)

X, y = dataset.drop(['samples', 'type'], axis=1), dataset['type'].map({'HCC': 1, 'normal': 0})
scaler = StandardScaler()

def to_tensor(x, y):
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y.values, dtype=torch.long)

class ImprovedCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim2, 2)
    
    def forward(self, x):
        if x.size(0) > 1:
            x = torch.relu(self.bn1(self.fc1(x)))
            x = self.drop1(x)
            x = torch.relu(self.bn2(self.fc2(x)))
            x = self.drop2(x)
        else:
            x = torch.relu(self.fc1(x))
            x = self.drop1(x)
            x = torch.relu(self.fc2(x))
            x = self.drop2(x)
        return self.fc3(x)

def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

def evaluate_model(model, loader):
    model.eval()
    all_targets, all_predictions = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_targets.extend(targets.numpy())
            all_predictions.extend(preds.numpy())
    return np.array(all_targets), np.array(all_predictions)

def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "ROC AUC": roc_auc_score(y_true, y_pred),
        "Confusion Matrix": cm.tolist()
    }

def process_parameter_set(params, X, y, skf, scaler):
    y_cv_true, y_cv_pred = [], []
    for train_idx, test_idx in skf.split(X, y):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

        X_train_cv_scaled = scaler.fit_transform(X_train_cv)
        X_test_cv_scaled = scaler.transform(X_test_cv)

        model = ImprovedCNN(
            input_dim=X_train_cv_scaled.shape[1],
            hidden_dim1=params['hidden_dim1'],
            hidden_dim2=params['hidden_dim2'],
            dropout_rate=params['dropout_rate']
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(TensorDataset(*to_tensor(X_train_cv_scaled, y_train_cv)), batch_size=128, shuffle=True)
        test_loader = DataLoader(TensorDataset(*to_tensor(X_test_cv_scaled, y_test_cv)), batch_size=128, shuffle=False)

        train_model(model, train_loader, criterion, optimizer)
        y_test_true, y_test_pred = evaluate_model(model, test_loader)

        y_cv_true.extend(y_test_true)
        y_cv_pred.extend(y_test_pred)

    metrics = compute_metrics(np.array(y_cv_true), np.array(y_cv_pred))
    metrics["params"] = params
    return metrics

param_grid = {
    'hidden_dim1': [64, 128],
    'hidden_dim2': [32, 64],
    'dropout_rate': [0.3, 0.5],
    'lr': [0.001, 0.0005],
    'weight_decay': [1e-4, 1e-5]
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
print("Starting parallel grid search...")

results = Parallel(n_jobs=4)(
    delayed(process_parameter_set)(params, X, y, skf, scaler)
    for params in ParameterGrid(param_grid)
)

# Save only the best result
best_result = max(results, key=lambda x: x["Recall"])

with open('best_fnn_metrics.json', 'w') as f:
    json.dump(best_result, f, indent=4)

print("Best metrics saved to best_fnn_metrics.json")
