import random
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from torch import Tensor
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class MWPDataset(Dataset):
    """
    This class implements the MWP dataset.
    """

    def __init__(self):
        self.mwps = list()
        self.equations = list()
        self.num_equations = list()
        self.operands = list()
        self.num_operands = list()
        self.operator_labels = list()
        self.vocabulary = list()

    def __len__(self) -> int:
        """
        This function returns the length of the dataset.
        Returns: The length of the dataset.

        """
        return len(self.mwps)

    def __getitem__(self, index: int) -> tuple[str, str, list[float], int, int, Tensor]:
        """
        This function returns the item at the given index.
        Args:
            index: The index of the item to return.

        Returns: The items at the given index.

        """
        return self.mwps[index], self.equations[index], self.operands[index], self.num_equations[index], \
            self.num_operands[index], self.operator_labels[index]

    def compute_vocabulary(self, min_count: int = 1) -> list[str]:
        """
        This function computes the vocabulary of the dataset.
        Args:
            min_count: The minimum count of a word to be included in the vocabulary.

        Returns: The vocabulary of the dataset.

        """
        vocabulary = Counter()
        for mwp in tqdm(self.mwps, desc="Computing vocabulary"):
            vocabulary.update(word_tokenize(mwp))
        vocabulary = [word for word, count in vocabulary.items() if count >= min_count]
        return vocabulary

    def get_batch(self, index, batch_size) -> tuple[
        list[str], list[str], list[list[float]], list[int], list[int], list[Tensor]]:
        """
        This function returns a batch of the dataset.
        Args:
            index: The index of the batch.
            batch_size: The size of the batch.

        Returns: A batch of the dataset.

        """
        return self.mwps[index:index + batch_size], \
            self.equations[index:index + batch_size], \
            self.operands[index:index + batch_size], \
            self.num_equations[index:index + batch_size], \
            self.num_operands[index:index + batch_size], \
            self.operator_labels[index:index + batch_size]

    @staticmethod
    def get_operator_labels(equations: list[str]) -> list[Tensor]:
        """
        This function returns the operator labels from the equations.
        Args:
            equations: The equations.

        Returns: The operator labels.

        """
        labels = list()
        position_map = {"+": 0, "-": 1, "*": 2, "/": 3}
        for equation in equations:
            current_label = [0.0, 0.0, 0.0, 0.0]
            if "+" in equation:
                current_label[position_map["+"]] = 1.0
            if "-" in equation:
                current_label[position_map["-"]] = 1.0
            if "*" in equation:
                current_label[position_map["*"]] = 1.0
            if "/" in equation:
                current_label[position_map["/"]] = 1.0
            labels.append(torch.tensor(current_label))
        return labels

    def load(self, path):
        """
        This function loads the dataset from the given path.
        Args:
            path: The path to load the dataset from.

        """
        if not Path(path).exists():
            raise FileNotFoundError
        df = pd.read_csv(path)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df.dropna(inplace=True)

        mwps = df["Question"].to_list()
        equations = df["Equation"].to_list()
        operands = df["Numbers"].to_list()
        assert len(mwps) == len(equations) == len(operands)

        self.mwps += mwps
        self.equations += equations
        self.num_equations += list(map(lambda x: len(x.split("<BRG>")), equations))
        self.operands += list(map(str.split, operands))
        self.num_operands += list(map(len, list(map(str.split, operands))))
        self.operator_labels += self.get_operator_labels(equations)

        assert len(self.mwps) == len(self.equations) == len(self.num_equations) == len(self.operands) == \
               len(self.num_operands) == len(self.operator_labels)

        self.vocabulary = self.compute_vocabulary()

    def unload(self):
        """
        This function unloads the dataset.

        """
        self.mwps = list()
        self.equations = list()
        self.num_equations = list()
        self.operands = list()
        self.num_operands = list()
        self.operator_labels = list()
        self.vocabulary = list()

    def shuffle(self):
        """
        This function shuffles the dataset.

        """
        zipped = list(
            zip(self.mwps, self.equations, self.operands, self.num_equations, self.num_operands, self.operator_labels))
        random.shuffle(zipped)
        self.mwps, self.equations, self.operands, self.num_equations, self.num_operands, self.operator_labels = zip(
            *zipped)
