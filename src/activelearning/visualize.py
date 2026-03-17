import matplotlib.pyplot as plt

def plot_learning_curves(acc_random, acc_us, acc_qbc_dict):
    plt.figure(figsize=(10,6))

    # Iterations axis
    iterations = range(1, len(acc_random)+1)

    # Plot random
    plt.plot(iterations, acc_random, label="Random", marker='o')

    # Plot uncertainty
    plt.plot(iterations, acc_us, label="Uncertainty", marker='s')

    # Plot QBC for hver committee size
    for n_comm, acc in acc_qbc_dict.items():
        plt.plot(iterations, acc, label=f"QBC {n_comm} members", marker='^')

    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Active Learning Performance")
    plt.legend()
    plt.grid(True)
    plt.show()