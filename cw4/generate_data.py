from typing import Optional

import numpy as np
import pandas as pd
from constants import CUR_DIR, TABLE_DIR, TRIES
from id3 import Id3


def format_float(number: float) -> str:
    return f"{number:.2f}".replace(".", ",")


def format_percent(number: float) -> str:
    number *= 100
    return f"{number:.2f}%".replace(".", ",")


def gen_confusion_csv(
    matrix: np.ndarray, filename: str, class_labels: list[str]
) -> None:
    vectorized_conversion = np.vectorize(format_float)
    formatted_matrix = vectorized_conversion(matrix)

    columns = pd.MultiIndex.from_tuples(
        [("Klasa rzeczywista", class_labels[0]), ("Klasa rzeczywista", class_labels[1])]
    )

    df = pd.DataFrame(
        formatted_matrix,
        index=pd.MultiIndex.from_tuples(
            [
                ("Klasa predykowana", class_labels[0]),
                ("Klasa predykowana", class_labels[1]),
            ]
        ),
        columns=columns,
    )
    df.to_csv(TABLE_DIR / filename)


def gen_accuracy_csv(accuracy_list: np.ndarray, filename: str) -> None:
    vectorized_conversion = np.vectorize(format_percent)

    results = np.array(
        [
            accuracy_list.min(),
            accuracy_list.mean(),
            accuracy_list.std(),
            accuracy_list.max(),
        ]
    )

    formatted_results: np.ndarray = vectorized_conversion(results)

    columns = ["min", "śr", "std", "max"]

    df = pd.DataFrame([formatted_results], columns=columns)

    df.to_csv(TABLE_DIR / filename, index=False)


def calculate_accuracy(test_data: pd.DataFrame, sample: pd.DataFrame) -> float:
    col1, col2 = test_data.iloc[:, 0], sample.iloc[:, 0]
    correct_predictions = (col1 == col2).sum()
    accuracy = correct_predictions / len(test_data)
    return accuracy


def gen_confusion_matrix(
    test_data: pd.DataFrame, sample: pd.DataFrame, classes: list[str]
):
    mistake_matrix = np.zeros((2, 2), dtype="int64")

    predicted_classes = sample[0]
    actual_classes = test_data[0]

    for predicted_class, actual_class in zip(predicted_classes, actual_classes):
        if predicted_class == actual_class:
            if predicted_class == classes[0]:
                mistake_matrix[0, 0] += 1
            else:
                mistake_matrix[1, 1] += 1
        else:
            if predicted_class == classes[0]:
                mistake_matrix[0, 1] += 1
            else:
                mistake_matrix[1, 0] += 1

    return mistake_matrix


def generate_data(
    training_part: int,
    testing_part: int,
    data_filename: str,
    classes: list[str],
    included_cols: list[int],
    limit_rows: Optional[int] = None,
):
    data_file = CUR_DIR / data_filename
    data = pd.read_csv(data_file, header=None)

    if not included_cols:
        included_cols = list(range(len(data.columns)))
    elif 0 not in included_cols:
        included_cols = [0] + included_cols

    split_ratio: float = training_part / (testing_part + training_part)

    indices = np.array(list(range(0, len(data))))
    rng = np.random.default_rng()
    train_indices = rng.choice(indices, int(len(data) * split_ratio), replace=False)
    test_indices = np.setdiff1d(indices, train_indices)

    if limit_rows:
        train_row_count = int(split_ratio * limit_rows)
        train_indices = rng.choice(train_indices, train_row_count, replace=False)
        test_indices = rng.choice(
            test_indices, limit_rows - train_row_count, replace=False
        )

    train_data = data.iloc[train_indices, included_cols]
    test_data = data.iloc[test_indices, included_cols]

    id3 = Id3()
    id3.train(train_data, 0)
    # pprint.pprint(id3.tree)

    sample = test_data.drop(test_data.columns[0], axis=1)

    sample = id3.predict(sample)

    return calculate_accuracy(test_data, sample), gen_confusion_matrix(
        test_data, sample, classes
    )


def generate_mushroom_data(
    filename: str,
    training_part: int,
    testing_part: int,
    included_cols: list[int] = [],
    tries: int = TRIES,
    limit_rows: Optional[int] = None,
):
    results = np.zeros(tries)
    mean_matrix = np.zeros((2, 2))

    for i in range(tries):
        accuracy, matrix = generate_data(
            training_part,
            testing_part,
            "mushroom.data",
            ["e", "p"],
            included_cols,
            limit_rows,
        )
        results[i] = accuracy
        mean_matrix += matrix

    mean_matrix /= tries

    gen_confusion_csv(
        mean_matrix,
        filename + "_mean_matrix.csv",
        ["Jadalny", "Niejadalny"],
    )
    gen_accuracy_csv(results, filename + "_accuracy.csv")


def generate_cancer_data(
    filename: str,
    training_part: int,
    testing_part: int,
    included_cols: list[int] = [],
    tries=TRIES,
    limit_rows: Optional[int] = None,
):
    results = np.zeros(tries)
    mean_matrix = np.zeros((2, 2))

    for i in range(tries):
        accuracy, matrix = generate_data(
            training_part,
            testing_part,
            "breast-cancer.data",
            ["no-recurrence-events", "recurrence-events"],
            included_cols,
            limit_rows,
        )
        results[i] = accuracy
        mean_matrix += matrix

    mean_matrix /= tries

    gen_confusion_csv(
        mean_matrix,
        filename + "_mean_matrix.csv",
        ["Brak nawrotu", "Nawrót"],
    )
    gen_accuracy_csv(results, filename + "_accuracy.csv")


if __name__ == "__main__":
    cols = list(range(23))
    cols.remove(5)
    # generate_mushroom_data("mushroom_5_20", 3, 2, [5, 20])
    # generate_mushroom_data("mushroom_20", 3, 2, [20])
    # generate_mushroom_data("mushroom_5", 3, 2, [5])
    # generate_mushroom_data("mushroom_22_3", 3, 2, [22, 3])
    # generate_mushroom_data("mushroom_full", 3, 2)
    # generate_mushroom_data("mushroom_-5", 3, 2, cols)
    # generate_mushroom_data("mushroom_200_rows", 3, 2, limit_rows=200)
    # generate_mushroom_data("mushroom_300_rows", 3, 2, limit_rows=300)
    generate_cancer_data("cancer_50_rows", 3, 2, limit_rows=50)
    # generate_cancer_data("cancer_full", 3, 2)
