from pathlib import Path

import numpy as np
import pandas as pd
from id3 import Id3

CUR_DIR = Path(__file__).parent


def calculate_accuracy(test_data: pd.DataFrame, sample: pd.DataFrame):
    col1, col2 = test_data.iloc[:, 0], sample.iloc[:, 0]
    correct_predictions = (col1 == col2).sum()
    accuracy = correct_predictions / len(test_data)
    return accuracy


def gen_mistake_matrix(
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


def generate_data(training_part: int, testing_part: int, data_filename: str):
    data_file = CUR_DIR / data_filename
    data = pd.read_csv(data_file, header=None)

    split_ratio: float = training_part / (testing_part + training_part)

    indices = np.arange(len(data))
    rng = np.random.default_rng()
    train_indices = rng.choice(indices, int(len(data) * split_ratio), replace=False)
    test_indices = np.setdiff1d(indices, train_indices)

    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]

    id3 = Id3()
    id3.train(train_data, 0)

    sample = test_data.drop(test_data.columns[0], axis=1)

    sample = id3.predict(sample)

    return calculate_accuracy(test_data, sample)
    print(gen_mistake_matrix(test_data, sample, np.unique(sample[0])))


if __name__ == "__main__":
    total = 0

    for i in range(100):
        x = generate_data(3, 2, "breast-cancer.data")
        total += x
        print(x)

    print(total / 100)

    # id3 = Id3()
    # id3.train(data, 0)

    # random_rows = data.sample(n=1000, random_state=42)
    # sample = random_rows.drop(random_rows.columns[0], axis=1)

    # sample = id3.predict(sample)
    # # print(sample)

    # # print(random_rows)

    # differences = random_rows.compare(sample)

    # print(differences)
