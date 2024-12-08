import os
import urllib.request
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Download dataset
url = "https://sbcb.inf.ufrgs.br/data/cumida/Genes/Liver/GSE14520_U133A/Liver_GSE14520_U133A.csv"
path = 'Liver_GSE14520_U133A.csv'
if not os.path.exists(path):
    urllib.request.urlretrieve(url, path)

# Load data
data = pd.read_csv(path)
if not {'samples', 'type'}.issubset(data.columns):
    exit()

# Encode labels
encoder = LabelEncoder()
data['type'] = encoder.fit_transform(data['type'])

# Prepare data
X = data.drop(['samples', 'type'], axis=1)
y = data['type']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature selection
selector = SelectFromModel(RandomForestClassifier(random_state=42), threshold="mean")
X_train_sel = selector.fit_transform(X_train, y_train)
X_val_sel = selector.transform(X_val)

# Model pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('clf', GaussianNB())])
params = {'clf__var_smoothing': [1e-9, 1e-8, 1e-7]}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

# Train model
grid = GridSearchCV(pipe, params, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(X_train_sel, y_train)

# Evaluate
y_pred = grid.predict(X_val_sel)
y_prob = grid.predict_proba(X_val_sel)[:, 1]
acc = (y_pred == y_val).mean()
roc_auc = roc_auc_score(y_val, y_prob)
report = classification_report(y_val, y_pred, target_names=encoder.classes_)
conf_matrix = confusion_matrix(y_val, y_pred)

print(f"Acurácia: {acc:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")
print("Relatório de Classificação:\n", report)
print("Matriz de Confusão:\n", conf_matrix)

# Plot ROC
fpr, tpr, _ = roc_curve(y_val, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save model and selector
joblib.dump(grid.best_estimator_, 'best_model.pkl')
joblib.dump(selector, 'feature_selector.pkl')
