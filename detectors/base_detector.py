from abc import ABC, abstractmethod

import numpy as np

from detectors.ci_tests import ci_test


class BlanketDetector(ABC):
    """A method that finds the Markov blanket of the target among the feature columns.

    A detector gives every feature column of a table a score, higher meaning more likely in the
    blanket (a ranking, evaluated without a threshold), and selects a set of columns (evaluated
    as a set). The classical detectors read the rows the models get as context, `X_train` and
    `y_train`, so that every method sees the same data.

    Attributes:
        name (str): A short name, used in the benchmark's tables and figures.
    """

    name = ""

    @abstractmethod
    def scores(self, dataset):
        """The score of every feature column.

        Args:
            dataset (CustomSCMDataset): A table with a known causal graph (simulated or a
                Causal Chamber).

        Returns:
            np.ndarray: One score per column of `dataset.X_train`.
        """

    def select(self, dataset):
        """The columns the detector puts in the blanket. By default, the columns above the
        largest gap of the scores (see `largest_gap`).

        Returns:
            np.ndarray: A boolean mask over the columns.
        """
        return largest_gap(self.scores(dataset))


def largest_gap(scores, floor=1e-6):
    """The columns above the largest ratio between two consecutive sorted scores.

    A threshold for a detector that only ranks: the blanket columns should stand apart from the
    others by a jump in score, wherever it lies. It always keeps at least one column.

    Args:
        scores (np.ndarray): The scores, higher is more likely in the blanket.
        floor (float, optional): Scores below it are raised to it, so that a ratio is defined.
            Defaults to 1e-6.

    Returns:
        np.ndarray: A boolean mask over the columns.
    """
    order = np.argsort(-scores)
    sorted_scores = np.maximum(scores[order], floor)
    if len(scores) < 2:
        return np.ones(len(scores), dtype=bool)
    cut = int(np.argmax(sorted_scores[:-1] / sorted_scores[1:])) + 1
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:cut]] = True
    return mask


class ConstraintBasedDetector(BlanketDetector):
    """A detector built on a conditional independence test of a column and the target (see
    detectors/ci_tests.py), fitted to the context rows of the table.

    The subclasses either find a blanket with an algorithm (`find_blanket`, then the score of a
    column is how strongly it depends on the target given the rest of the found blanket), or
    score every column by one test and select the significant ones.

    Args:
        test (str, optional): A key of CI_TESTS. Defaults to "logistic".
        alpha (float, optional): The level of the tests. Defaults to 0.05.
    """

    base_name = ""

    def __init__(self, test="logistic", alpha=0.05):
        self.test_name, self.alpha = test, alpha
        self.name = f"{self.base_name}:{test}"
        self._cache = {}  # table name -> (fitted test, found blanket)

    def fitted(self, dataset):
        """The test fitted to the context rows of the table, and the blanket found (None for a
        detector without an algorithm), computed once per table."""
        if dataset.name not in self._cache:
            test = ci_test(self.test_name, dataset.X_train, dataset.y_train)
            self._cache[dataset.name] = (test, self.find_blanket(test, dataset.X_train.shape[1]))
        return self._cache[dataset.name]

    def find_blanket(self, test, n_columns):
        """The columns found by the algorithm, None for a detector that only tests."""
        return None

    def scores(self, dataset):
        test, blanket = self.fitted(dataset)
        p = [test.pvalue(j, [k for k in blanket if k != j]) for j in range(dataset.X_train.shape[1])]
        return -np.log10(np.maximum(p, 1e-300))

    def select(self, dataset):
        test, blanket = self.fitted(dataset)
        mask = np.zeros(dataset.X_train.shape[1], dtype=bool)
        mask[list(blanket)] = True
        return mask

    def shrink(self, test, blanket):
        """Remove the columns independent of the target given the rest of the blanket, until none
        is (the backward phase of IAMB and Grow-Shrink, which removes the false positives)."""
        blanket = list(blanket)
        changed = True
        while changed:
            changed = False
            for j in list(blanket):
                if test.pvalue(j, [k for k in blanket if k != j]) > self.alpha:
                    blanket.remove(j)
                    changed = True
        return blanket
