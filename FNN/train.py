import os
import urllib.request
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, roc_auc_score,
    classification_report
)

def set_deterministic():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_deterministic()

# Dataset configuration
dataset_url = "https://sbcb.inf.ufrgs.br/data/cumida/Genes/Liver/GSE14520_U133A/Liver_GSE14520_U133A.csv"
file_path = 'Liver_GSE14520_U133A.csv'
if not os.path.exists(file_path):
    urllib.request.urlretrieve(dataset_url, file_path)
dataset = pd.read_csv(file_path)

# Data split
X, y = dataset.drop(['samples', 'type'], axis=1), dataset['type'].map({'HCC': 1, 'normal': 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
def to_tensor(x, y):
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y.values, dtype=torch.long)
X_train_tensor, y_train_tensor = to_tensor(X_train_scaled, y_train)
X_test_tensor, y_test_tensor = to_tensor(X_test_scaled, y_test)

# Datasets and DataLoader
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False)

# Improved CNN model
class ImprovedCNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 2)
    
    def forward(self, x):
        if x.size(0) > 1:  # Apply BatchNorm for batches larger than 1
            x = torch.relu(self.bn1(self.fc1(x)))
            x = self.drop1(x)
            x = torch.relu(self.bn2(self.fc2(x)))
            x = self.drop2(x)
        else:  # Skip BatchNorm for single batch samples
            x = torch.relu(self.fc1(x))
            x = self.drop1(x)
            x = torch.relu(self.fc2(x))
            x = self.drop2(x)
        x = self.fc3(x)
        return x

# Initialize model
model = ImprovedCNN(X_train_scaled.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs=20):
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
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# Evaluation function
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

# Compute metrics
def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "ROC AUC": roc_auc_score(y_true, y_pred),
        "Confusion Matrix": cm
    }

# Train the model
train_model(model, train_loader, criterion, optimizer)

# Evaluate performance
y_train_true, y_train_pred = evaluate_model(model, train_loader)
train_metrics = compute_metrics(y_train_true, y_train_pred)

# Cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
y_cv_true = []
y_cv_pred = []

for train_idx, test_idx in skf.split(X, y):
    X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
    y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

    X_train_cv_scaled = scaler.fit_transform(X_train_cv)
    X_test_cv_scaled = scaler.transform(X_test_cv)

    model = ImprovedCNN(X_train_cv_scaled.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    train_loader = DataLoader(TensorDataset(*to_tensor(X_train_cv_scaled, y_train_cv)), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(*to_tensor(X_test_cv_scaled, y_test_cv)), batch_size=64, shuffle=False)

    train_model(model, train_loader, criterion, optimizer, epochs=10)
    y_test_true, y_test_pred = evaluate_model(model, test_loader)

    y_cv_true.extend(y_test_true)
    y_cv_pred.extend(y_test_pred)

cv_metrics = compute_metrics(y_cv_true, y_cv_pred)

# Print results in the desired format
print("\n=== Training Metrics ===")
for metric, value in train_metrics.items():
    if metric == "Confusion Matrix":
        print(f"\nConfusion Matrix:\n{value}")
    else:
        print(f"{metric}: {value:.4f}")

print("\n=== Cross-Validation Metrics ===")
for metric, value in cv_metrics.items():
    if metric == "Confusion Matrix":
        print(f"\nConfusion Matrix:\n{value}")
    else:
        print(f"{metric}: {value:.4f}")

print("\n=== Detailed Classification Report ===")
report = classification_report(y_cv_true, y_cv_pred, target_names=['Normal', 'HCC'])
print(report)