import numpy as np
from sklearn.model_selection import train_test_split
from model import Model, random_sampling, uncertainty_sampling, qbc_sampling
from data import Xpool_all, ypool_all, Xtest, ytest
from visualize import plot_learning_curves

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
        model.fit(Xtrain, ytrain)
        ypred = model.predict(Xtest)
        acc = np.mean(ypred == ytest)
        accs.append(acc)

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


    # returnér kun sidste accuracy
    return accs



if __name__ == "__main__":
    seed = 42
    n_init = 10
    addn = 5
    n_comm_list = [3, 5, 10, 20]

    # Random sampling
    acc_random = active_learning(Xpool_all.copy(), ypool_all.copy(), Xtest, ytest,
                                strategy="random", n_init=n_init, addn=addn, seed=seed)
    print(f"Random sampling: acc = {acc_random[-1]:.3f}")

    # Uncertainty sampling
    acc_us = active_learning(Xpool_all.copy(), ypool_all.copy(), Xtest, ytest,
                            strategy="uncertainty", n_init=n_init, addn=addn, seed=seed)
    print(f"Uncertainty sampling: acc = {acc_us[-1]:.3f}")

    # QBC med forskellige committee sizes
    acc_qbc_dict = {}
    for n_comm in n_comm_list:
        acc_qbc_dict[n_comm] = active_learning(
            Xpool_all.copy(), ypool_all.copy(), Xtest, ytest,
            strategy="qbc", n_init=n_init, addn=addn, n_comm=n_comm, seed=seed
        )
    for n_comm, acc in acc_qbc_dict.items():
        print(f"QBC with {n_comm} committee members: acc = {acc[-1]:.3f}")
    plot_learning_curves(acc_random, acc_us, acc_qbc_dict)