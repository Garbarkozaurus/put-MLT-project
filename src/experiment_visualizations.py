import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def scatter_train_v_test(experiment_results: np.ndarray[np.float32]) -> None:
    train_acc = experiment_results[:, 1]
    test_acc = experiment_results[:, 2]
    reg_coefs = experiment_results[:, 0]
    plt.scatter(
        train_acc,
        test_acc,
        linestyle="None",
        marker="o",
        c=np.log2(reg_coefs),
        cmap=plt.get_cmap("viridis"),
        alpha=0.5,
    )
    plt.colorbar()
    plt.gca().set_axisbelow(True)
    plt.grid()
    plt.xlabel("Train accuracy")
    plt.ylabel("Test accuracy")
    plt.show()


def diff_train_v_test(experiment_results: np.ndarray[np.float32]) -> None:
    train_acc = experiment_results[:, 1]
    test_acc = experiment_results[:, 2]
    reg_coefs = experiment_results[:, 0]
    reg_coefs = np.unique(reg_coefs)
    diffs = train_acc - test_acc
    diffs = diffs.reshape((-1, len(reg_coefs)))
    # train_acc = train_acc.reshape((-1, len(reg_coefs)))
    # test_acc = test_acc.reshape((-1, len(reg_coefs)))
    # mean_train_acc = np.mean(train_acc, axis=0)
    # mean_test_acc = np.mean(test_acc, axis=0)
    mean_diffs = np.mean(diffs, axis=0)
    # plt.plot(np.log2(reg_coefs), mean_train_acc - mean_test_acc, linestyle="dashed", marker="o")
    # plt.errorbar(np.log2(reg_coefs), mean_train_acc - mean_test_acc, np.std(train_acc - test_acc, axis=0),  linestyle="dashed", marker="o", capsize=2)
    # plt.plot(np.log2(reg_coefs), mean_train_acc - mean_test_acc, linestyle="dashed", marker="o")
    plt.plot(np.log2(reg_coefs), mean_diffs, linestyle="dashed", marker="o")
    plt.gca().set_axisbelow(True)
    plt.xlabel("$\log_2$C")
    plt.axhline(0, c="grey")
    plt.ylabel("Train accuracy - Test accuracy")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    penguins = pd.read_csv("./data/penguins_results.csv", header=None).to_numpy()
    wine = pd.read_csv("./data/wine_results.csv", header=None).to_numpy()

    scatter_train_v_test(penguins)
    diff_train_v_test(penguins)
    scatter_train_v_test(wine)
    diff_train_v_test(wine)
