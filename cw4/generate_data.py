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


def generate_data(training_part: int, testing_part: int, data_filename: str):
    data_file = CUR_DIR / data_filename
    data = pd.read_csv(data_file, header=None)

    split_ratio: float = training_part / (testing_part + training_part)

    indices = np.array(list(range(0, len(data))))
    rng = np.random.default_rng()
    train_indices = rng.choice(indices, int(len(data) * split_ratio), replace=False)
    test_indices = np.setdiff1d(indices, train_indices)

    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]

    id3 = Id3()
    id3.train(train_data, 0)

    sample = test_data.drop(test_data.columns[0], axis=1)

    sample = id3.predict(sample)

    print(calculate_accuracy(test_data, sample))


if __name__ == "__main__":

    for i in range(100):
        generate_data(3, 2, "breast-cancer.data")

    # id3 = Id3()
    # id3.train(data, 0)

    # random_rows = data.sample(n=1000, random_state=42)
    # sample = random_rows.drop(random_rows.columns[0], axis=1)

    # sample = id3.predict(sample)
    # # print(sample)

    # # print(random_rows)

    # differences = random_rows.compare(sample)

    # print(differences)
