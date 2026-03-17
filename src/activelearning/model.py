import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from scipy.stats import entropy

# Defining logisitc regression model
class Model:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    

# DEFINING SAMPLING METHODS 

# Random sampling 
def random_sampling(Xtrain, ytrain, Xpool, ypool, addn):
    # træk tilfældige samples fra pool
    poolidx = np.arange(len(Xpool))
    idx = np.random.choice(poolidx, size=min(addn, len(poolidx)), replace=False)
    X_add = Xpool[idx]
    y_add = ypool[idx]
    # fjern fra pool
    Xpool = np.delete(Xpool, idx, axis=0)
    ypool = np.delete(ypool, idx, axis=0)
    # returnér de nye træningsdata og opdateret pool
    Xtrain = np.concatenate([Xtrain, X_add], axis=0)
    ytrain = np.concatenate([ytrain, y_add], axis=0)
    return Xtrain, ytrain, Xpool, ypool

# Uncertainty sampling 
def uncertainty_sampling(Xtrain, ytrain, Xpool, ypool, model, addn):

    model.fit(Xtrain, ytrain)
    
    # evaluer probabilities på pool
    probs = model.predict_proba(Xpool)
    max_probs = np.max(probs, axis=1)
    
    # vælg least confident samples
    least_conf_idx = np.argsort(max_probs)[:addn]
    
    Xtrain = np.concatenate([Xtrain, Xpool[least_conf_idx]], axis=0)
    ytrain = np.concatenate([ytrain, ypool[least_conf_idx]], axis=0)
    
    Xpool = np.delete(Xpool, least_conf_idx, axis=0)
    ypool = np.delete(ypool, least_conf_idx, axis=0)
    
    return Xtrain, ytrain, Xpool, ypool

def qbc_sampling(Xtrain, ytrain, Xpool, ypool, addn, n_comm=10, seed=42):
    np.random.seed(seed)
    
    # 1. Kør committee members
    model_preds = []
    for j in range(n_comm):
        boot_x, boot_y = resample(Xtrain, ytrain, stratify=ytrain, random_state=seed+j)
        model = Model()
        model.fit(boot_x, boot_y)
        preds = model.predict(Xpool)
        model_preds.append(preds)
    
    model_preds = np.array(model_preds)
    
    # 2. Beregn vote entropy
    vote_entropy = []
    for k in range(model_preds.shape[1]):
        counts = np.bincount(model_preds[:, k])
        probs = counts / counts.sum()
        vote_entropy.append(entropy(probs))
    
    vote_entropy = np.array(vote_entropy)
    
    # 3. Vælg mest uafklarede samples
    q_idx = np.argsort(-vote_entropy)[:addn]
    
    # 4. Tilføj til træning og fjern fra pool
    Xtrain = np.concatenate([Xtrain, Xpool[q_idx]], axis=0)
    ytrain = np.concatenate([ytrain, ypool[q_idx]], axis=0)
    
    Xpool = np.delete(Xpool, q_idx, axis=0)
    ypool = np.delete(ypool, q_idx, axis=0)
    
    return Xtrain, ytrain, Xpool, ypool



if __name__ == "__main__":
    model = Model()
