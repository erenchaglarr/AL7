# train.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from data import *
from model import Model, random_sampling, uncertainty_sampling, qbc_sampling
from data import Xpool_all, ypool_all, Xtest, ytest


# Active Learning loop
def active_learning(Xpool, ypool, Xtest, ytest, strategy="random",
                    n_init=10, addn=5, n_comm=5, seed=42):
    np.random.seed(seed)

    # initial training set
    poolidx = np.arange(len(Xpool))
    train_idx = np.random.choice(poolidx, size=n_init, replace=False)
    
    Xtrain = Xpool[train_idx]
    ytrain = ypool[train_idx]
    
    # fjern dem fra pool
    Xpool_remain = np.delete(Xpool, train_idx, axis=0)
    ypool_remain = np.delete(ypool, train_idx, axis=0)

    accs = []

    for i in range(25):
        model = Model()
        
        # Eval på testset
        model.fit(Xtrain, ytrain)
        ypred = model.predict(Xtest)
        acc = np.mean(ypred == ytest)
        accs.append((len(Xtrain), acc))
        print(f"{strategy}, iteration {i+1}, training samples {len(Xtrain)}, acc {acc:.3f}")

        if len(Xpool_remain) == 0:
            break

        # Sampling
        if strategy == "random":
            Xtrain, ytrain, Xpool_remain, ypool_remain = random_sampling(
                Xtrain, ytrain, Xpool_remain, ypool_remain, addn
            )
        elif strategy == "uncertainty":
            Xtrain, ytrain, Xpool_remain, ypool_remain = uncertainty_sampling(
                Xtrain, ytrain, Xpool_remain, ypool_remain, model, addn
            )
        elif strategy == "qbc":
            Xtrain, ytrain, Xpool_remain, ypool_remain = qbc_sampling(
                Xtrain, ytrain, Xpool_remain, ypool_remain, addn, n_comm=n_comm, seed=seed
            )
        else:
            raise ValueError("Unknown strategy")

    return accs


# Main: Load data + kør AL
if __name__ == "__main__":
    seed = 42
    n_init = 10
    addn = 5
    n_comm_list = [3, 5, 10]

    # Random sampling
    acc_random = active_learning(Xpool_all.copy(), ypool_all.copy(), Xtest, ytest,
                                 strategy="random", n_init=n_init, addn=addn, seed=seed)

    # Uncertainty sampling
    acc_us = active_learning(Xpool_all.copy(), ypool_all.copy(), Xtest, ytest,
                             strategy="uncertainty", n_init=n_init, addn=addn, seed=seed)

    # QBC med forskellig committee size
    for n_comm in n_comm_list:
        acc_qbc = active_learning(Xpool_all.copy(), ypool_all.copy(), Xtest, ytest,
                                  strategy="qbc", n_init=n_init, addn=addn,
                                  n_comm=n_comm, seed=seed)