from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import data_loading
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

from typing import Literal


def regularization_experiment(
    dataset: Literal["penguins"] | Literal["wine"],
    model_type: Literal["linear"] | Literal["rbf"],
    regularization_coefficients: np.ndarray[np.float32],
    data_split_seed: int = 0,
    verbose: bool = False,
) -> np.ndarray[np.float32]:
    match dataset:
        case "penguins":
            df = data_loading.load_penguins()
            train_df, test_df = train_test_split(
                df, stratify=df["species"], random_state=data_split_seed
            )
        case "wine":
            df = data_loading.load_wine()
            train_df, test_df = train_test_split(
                df, stratify=df["quality"], random_state=data_split_seed
            )
        case _:
            raise ValueError(
                f"Incorrect dataset name: '{dataset}'. Expected one of: 'penguins', 'wine'"
            )
    x_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    x_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    accuracies = np.zeros((len(regularization_coefficients), 2))
    for i, r_coef in enumerate(regularization_coefficients):
        match model_type:
            case "linear":
                model = LinearSVC(C=r_coef, max_iter=100000)
            case "rbf":
                model = SVC(C=r_coef)
            case _:
                raise ValueError(
                    f"Incorrect model type: '{model_type}'. Expected one of: 'linear'"
                )
        if verbose:
            print(
                f"{datetime.now().strftime('%H:%M:%S')}|{dataset}|{r_coef} ({i+1}/{len(regularization_coefficients)})"
            )
        model.fit(x_train, y_train)
        train_predictions = model.predict(x_train)
        test_predictions = model.predict(x_test)
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        accuracies[i] = train_accuracy, test_accuracy
    regularization_coefficients = regularization_coefficients.reshape(-1, 1)
    return np.hstack((regularization_coefficients, accuracies))


def multiple_experiments(
    dataset: Literal["penguins"] | Literal["wine"],
    model_type: Literal["linear"] | Literal["rbf"],
    num_runs: int = 100,
) -> None:
    DEFAULT_REGULARIZATION_COEFFICIENTS = np.array([2**x for x in range(-10, 11)])
    with open(f"{dataset}_results.csv", "a+") as fp:
        for i in range(num_runs):
            print(f"STARTING RUN {i+1}")
            results = regularization_experiment(
                dataset, model_type, DEFAULT_REGULARIZATION_COEFFICIENTS, i
            )
            for reg, train, test in results:
                fp.write(f"{reg},{train},{test}\n")


if __name__ == "__main__":
    reg_coefs = np.array([2**x for x in range(-10, 11)])
    ret = regularization_experiment("penguins", "linear", reg_coefs, verbose=True)
    print(ret)
