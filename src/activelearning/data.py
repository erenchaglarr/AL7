from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Hent datasæt
mushroom = fetch_ucirepo(id=73)

X = mushroom.data.features
y = mushroom.data.targets.squeeze()

# One-hot encode features
try:
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)

X_encoded = enc.fit_transform(X)

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

# Brug kun 10% af poolen til active learning
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