import os
import urllib.request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Download dataset
url = "https://sbcb.inf.ufrgs.br/data/cumida/Genes/Liver/GSE14520_U133_2/Liver_GSE14520_U133_2.csv"
path = 'Liver_GSE14520_U133_2.csv'
if not os.path.exists(path):
    urllib.request.urlretrieve(url, path)

# Load data and model
data = pd.read_csv(path)
model = joblib.load('best_model.pkl')
selector = joblib.load('feature_selector.pkl')

# Encode labels
encoder = LabelEncoder()
encoder.fit(['HCC', 'normal'])

# Prepare data
X = selector.transform(data.drop(['samples', 'type'], axis=1))
y = encoder.transform(data['type'])

# Predict and evaluate
try:
    preds = model.predict(X)
    acc = (preds == y).mean()
    report = classification_report(y, preds, target_names=encoder.classes_)
    conf_matrix = confusion_matrix(y, preds)

    print(f"\nAccuracy: {acc:.2f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(conf_matrix)
except ValueError as e:
    print(f"Error: {e}")
