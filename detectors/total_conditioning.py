import numpy as np

from detectors.base_detector import ConstraintBasedDetector


class TotalConditioning(ConstraintBasedDetector):
    """Total conditioning (Pellet & Elisseeff, JMLR 2008): a column is in the blanket iff it
    depends on the target given every other column. One test per column, no search.

    Its score is the strength of that test, which makes it the classical counterpart of the
    perturbation of one column in a model that reads all the others.
    """

    base_name = "total_conditioning"

    def scores(self, dataset):
        test, _ = self.fitted(dataset)
        n = dataset.X_train.shape[1]
        p = [test.pvalue(j, [k for k in range(n) if k != j]) for j in range(n)]
        return -np.log10(np.maximum(p, 1e-300))

    def select(self, dataset):
        return self.scores(dataset) >= -np.log10(self.alpha)
