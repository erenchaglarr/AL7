from torch import nn
import torch

class Model(nn.Module):
    """Just a dummy model to show how to structure your code"""
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
    

# QBC
from scipy.stats import entropy
Xpool = np.concatenate(X[:14])[:, mask.ravel()]
ypool = np.concatenate(y[:14])
testacc_qbc=[]
ncomm=10
trainset=order[:ninit]
Xtrain=np.take(Xpool,trainset,axis=0)
ytrain=np.take(ypool,trainset,axis=0)
poolidx=np.arange(len(Xpool),dtype=int)
poolidx=np.setdiff1d(poolidx,trainset)
for i in range(25):
    model_preds = []
    for j in range(ncomm):
        boot_x, boot_y = sklearn.utils.resample(Xtrain, ytrain, stratify=ytrain)
        model = lin.LogisticRegression(C=1, max_iter=1000)
        model.fit(boot_x, boot_y)
        preds = model.predict(Xpool)
        model_preds.append(preds)
    
    model_preds = np.array(model_preds)
    vote_entropy = []

    for k in range(model_preds.shape[1]): 
        votes = model_preds[:, k] 
        
        counts = np.bincount(votes) 
        probs = counts / counts.sum()  
        
        vote_entropy.append(entropy(probs))
        
    vote_entropy = np.array(vote_entropy)
    # choose most uncertain samples
    q_idx = np.argsort(-vote_entropy)[:addn]

    # evaluate model on test set
    model = lin.LogisticRegression(C=1, max_iter=1000)
    model.fit(Xtrain, ytrain)

    ypred = model.predict(Xtest)
    acc = np.mean(ypred == ytest)

    testacc_qbc.append((len(Xtrain), acc))

    print('QBC, training samples:', len(Xtrain), 'accuracy:', acc)

    # add selected samples to training set
    Xtrain = np.concatenate([Xtrain, Xpool[q_idx]], axis=0)
    ytrain = np.concatenate([ytrain, ypool[q_idx]], axis=0)

    # remove them from pool
    Xpool = np.delete(Xpool, q_idx, axis=0)
    ypool = np.delete(ypool, q_idx, axis=0)

# Uncertainty sampling 
testacc_al=[]
trainset=order[:ninit]
Xtrain=np.take(Xpool,trainset,axis=0)
ytrain=np.take(ypool,trainset,axis=0)
poolidx=np.arange(len(Xpool),dtype=int)
poolidx=np.setdiff1d(poolidx,trainset)
for i in range(25):
    model.fit(Xtrain, ytrain)

    ypred = model.predict(Xtest)
    acc = np.mean(ypred == ytest)
    testacc_al.append((len(Xtrain), acc))
    
    # Least confident sampling
    probs = model.predict_proba(Xpool)
    max_probs = np.max(probs, axis=1)
    least_conf_idx = np.argsort(max_probs)[:addn]

    Xtrain = np.concatenate([Xtrain, Xpool[least_conf_idx]], axis=0)
    ytrain = np.concatenate([ytrain, ypool[least_conf_idx]], axis=0)

    Xpool = np.delete(Xpool, least_conf_idx, axis=0)
    ypool = np.delete(ypool, least_conf_idx, axis=0)



if __name__ == "__main__":
    model = Model()
    x = torch.rand(1)
    print(f"Output shape of model: {model(x).shape}")
