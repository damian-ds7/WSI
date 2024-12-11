from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class Id3:
    def __init__(self) -> None:
        self.tree = None
        self.most_common_class = None

    @staticmethod
    def entropy(y: pd.DataFrame) -> float:
        frequency: Counter[str] = Counter(y)
        total_instances: int = len(y)
        return -np.sum(
            (count / total_instances) * np.log2(count / total_instances)
            for count in frequency.values()
        )

    def information_gain(
        self, dataset: pd.DataFrame, current_attribute: int, target_attribute: int
    ) -> float:
        total_entropy = self.entropy(dataset[target_attribute])

        values, counts = np.unique(dataset[current_attribute], return_counts=True)
        subset_entropy: float = 0

        for val, count in zip(values, counts):
            subset = dataset.where(dataset[current_attribute] == val)
            subset = subset.dropna()[target_attribute]

            subset_entropy += count / np.sum(counts) * self.entropy(subset)

        return total_entropy - subset_entropy

    def _id3(
        self,
        dataset: pd.DataFrame,
        target_index: int,
        attributes: NDArray[str] = None,
        parent_class: int = None,
    ) -> dict:
        tree = {}

        if attributes is None:
            attributes = np.array(dataset.columns)
            attributes = np.delete(attributes, target_index)

        if len(rem_classes := np.unique(dataset[target_index])) == 1:
            return rem_classes[0]

        elif len(dataset) == 0:
            return parent_class

        elif len(attributes) == 0:
            return dataset[target_index].mode()[0]

        else:
            parent_class = dataset[target_index].mode()
            parent_class = parent_class[0]
            info_gains = np.array(
                [
                    self.information_gain(dataset, attr, target_index)
                    for attr in attributes
                ]
            )

            best_attribute_index = np.argmax(info_gains)
            best_attribute = attributes[best_attribute_index]

            tree = {best_attribute: {}}

            attributes = np.delete(attributes, best_attribute_index)

            for value in np.unique(dataset[best_attribute]):
                subset = dataset.where(dataset[best_attribute] == value).dropna()
                subtree = self._id3(subset, target_index, attributes, parent_class)
                tree[best_attribute][value] = subtree

        return tree

    def train(self, dataset: pd.DataFrame, target_index: int) -> None:
        self.tree = self._id3(dataset, target_index)

        targets = dataset[target_index]
        self.most_common_class = Counter(targets).most_common()[0][0]

    def _predict(self, tree: dict, sample: pd.Series) -> str:
        if not isinstance(tree, dict):
            return tree

        root_attribute: str = list(tree.keys())[0]
        attribute_value = sample[root_attribute]

        subtree = tree[root_attribute].get(attribute_value, self.most_common_class)

        if subtree is None:
            return None

        return self._predict(subtree, sample)

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if self.tree is None:
            raise ValueError("Decision tree doesn't exist, classifier not trained")

        predicted_classes: NDArray[np.int64] = np.empty(len(dataset), dtype=object)

        for i, (_, row) in enumerate(dataset.iterrows()):
            predicted_class = self._predict(self.tree, row)
            predicted_classes[i] = predicted_class

        predicted_classes_series = pd.Series(predicted_classes, index=dataset.index)
        dataset = pd.concat([predicted_classes_series, dataset], axis=1)

        return dataset


if __name__ == "__main__":
    cur_dir = Path(__file__).parent
    data_file = cur_dir / "mushroom.data"
    data = pd.read_csv(data_file, header=None)  # Load without header

    id3 = Id3()
    id3.train(data, 0)

    random_rows = data.sample(n=1000, random_state=42)
    sample = random_rows.drop(random_rows.columns[0], axis=1)

    sample = id3.predict(sample)
    # print(sample)

    # print(random_rows)

    differences = random_rows.compare(sample)

    print(differences)
