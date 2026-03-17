from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Hent Adult / Census Income fra OpenML
adult = fetch_openml(name="adult", version=2, as_frame=True)

X = adult.data.copy()
y = adult.target.copy()

# Håndter missing values robust
for col in X.columns:
    if pd.api.types.is_categorical_dtype(X[col]):
        X[col] = X[col].cat.add_categories(["missing"]).fillna("missing")
    elif X[col].dtype == "object":
        X[col] = X[col].fillna("missing")
    else:
        X[col] = X[col].fillna(X[col].median())

# One-hot encode alle kategoriske features
X_encoded = pd.get_dummies(X, drop_first=False)
X_encoded = X_encoded.astype(float).to_numpy()

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Fast test-sæt
Xpool_full, Xtest, ypool_full, ytest = train_test_split(
    X_encoded,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Brug kun en mindre del af poolen til active learning
Xpool_all, _, ypool_all, _ = train_test_split(
    Xpool_full,
    ypool_full,
    train_size=0.10,
    random_state=42,
    stratify=ypool_full
)

print("Xpool_all:", Xpool_all.shape)
print("ypool_all:", ypool_all.shape)
print("Xtest:", Xtest.shape)
print("ytest:", ytest.shape)