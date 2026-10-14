import numpy as np

from detectors.base_detector import ConstraintBasedDetector


class Marginal(ConstraintBasedDetector):
    """Not a Markov blanket method: the columns associated with the target, each tested alone.

    The reference a blanket detector must beat. A column outside the blanket that shares a cause
    with the target (another sensor lit by the same light source) is associated with it and
    selected here, though it is useless once the blanket is known.
    """

    base_name = "marginal"

    def scores(self, dataset):
        test, _ = self.fitted(dataset)
        p = [test.pvalue(j, []) for j in range(dataset.X_train.shape[1])]
        return -np.log10(np.maximum(p, 1e-300))

    def select(self, dataset):
        return self.scores(dataset) >= -np.log10(self.alpha)
