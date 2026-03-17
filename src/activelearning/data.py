from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Hent datasæt
mushroom = fetch_ucirepo(id=73)

X = mushroom.data.features
y = mushroom.data.targets

# One-hot encode X
enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = enc.fit_transform(X)

# Gør y til 1D
y = y.squeeze()   # hvis y er en DataFrame med én kolonne

# Encode labels (fx edible/poisonous -> 0/1)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("X_encoded shape:", X_encoded.shape)
print("y_encoded shape:", y_encoded.shape)

# 1) Del først i pool + test
Xpool_all, Xtest, ypool_all, ytest = train_test_split(
    X_encoded,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Xpool_all:", Xpool_all.shape)
print("ypool_all:", ypool_all.shape)
print("Xtest:", Xtest.shape)
print("ytest:", ytest.shape)

# 2) Vælg et lille initialt træningssæt fra poolen
ninit = 20   # antal start-labels
rng = np.random.default_rng(42)
order = rng.permutation(len(Xpool_all))

trainset = order[:ninit]

Xtrain = Xpool_all[trainset]
ytrain = ypool_all[trainset]

# Resten bliver unlabeled pool
poolidx = np.setdiff1d(np.arange(len(Xpool_all)), trainset)
Xpool = Xpool_all[poolidx]
ypool = ypool_all[poolidx]

print("Xtrain:", Xtrain.shape)
print("ytrain:", ytrain.shape)
print("Xpool:", Xpool.shape)
print("ypool:", ypool.shape)
