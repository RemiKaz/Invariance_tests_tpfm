"""Tests of the conditional independence of a feature column and the target, x_j _||_ y | X_S.

They are what the classical Markov blanket algorithms are made of. The target is a class
index; the features are continuous.

- `fisher_z`: partial correlation of x_j and y (the class index read as a number) given X_S,
  with Fisher's z. Linear and Gaussian: the textbook test of IAMB and Grow-Shrink.
- `logistic`: likelihood-ratio test of a multinomial logistic regression of y on X_S against
  the same with x_j added. Linear in the features, but made for a class target.
- `spline`: the same with every feature entering through a cubic spline, to catch non-linear
  and non-monotone effects (at the price of power on small tables).
"""

import numpy as np
from scipy.stats import chi2, norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler


class FisherZ:
    """Partial correlation of x_j and y given X_S, Fisher's z test."""

    name = "fisher_z"

    def __init__(self, X, y):
        self.X = StandardScaler().fit_transform(X)
        self.y = y.astype(float)
        self.n = len(y)

    def pvalue(self, j, S):
        """The p-value of x_j _||_ y | X_S.

        Args:
            j (int): The column tested.
            S (list): The columns conditioned on.

        Returns:
            float: The p-value.
        """
        Z = np.column_stack([np.ones(self.n)] + [self.X[:, k] for k in S])
        rx = self.X[:, j] - Z @ np.linalg.lstsq(Z, self.X[:, j], rcond=None)[0]
        ry = self.y - Z @ np.linalg.lstsq(Z, self.y, rcond=None)[0]
        if rx.std() < 1e-12 or ry.std() < 1e-12:
            return 1.0
        r = np.clip(np.corrcoef(rx, ry)[0, 1], -0.999999, 0.999999)
        z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(max(self.n - len(S) - 3, 1))
        return float(2 * norm.sf(abs(z)))


class LogisticLR:
    """Likelihood-ratio test of y ~ X_S against y ~ X_S + x_j, multinomial logistic models.

    The fitted log-likelihoods are cached by conditioning set: the algorithms ask for the same
    sets again and again.
    """

    name = "logistic"

    def __init__(self, X, y, spline=False):
        self.y = np.unique(y, return_inverse=True)[1]
        self.n, self.n_classes = len(y), int(self.y.max()) + 1
        X = StandardScaler().fit_transform(X)
        if spline:
            self.name = "spline"
            self.basis = [SplineTransformer(n_knots=4, degree=3, include_bias=False).fit_transform(X[:, [k]])
                          for k in range(X.shape[1])]
        else:
            self.basis = [X[:, [k]] for k in range(X.shape[1])]
        self._loglik = {}

    def loglik(self, S):
        """The log-likelihood of the logistic regression of y on the columns `S`."""
        key = frozenset(S)
        if key not in self._loglik:
            if not S:
                p = np.bincount(self.y) / self.n
                value = np.log(p[self.y]).sum()
            else:
                Z = np.hstack([self.basis[k] for k in sorted(S)])
                # a large C rather than no penalty: the MLE diverges when a column separates the
                # classes (a sensor that measures the target), the likelihood ratio does not
                model = LogisticRegression(C=1e4, max_iter=2000).fit(Z, self.y)
                proba = model.predict_proba(Z)[np.arange(self.n), self.y]
                value = np.log(np.clip(proba, 1e-12, None)).sum()
            self._loglik[key] = float(value)
        return self._loglik[key]

    def pvalue(self, j, S):
        """The p-value of x_j _||_ y | X_S, from the likelihood-ratio statistic."""
        S = [k for k in S if k != j]
        statistic = 2 * (self.loglik(S + [j]) - self.loglik(S))
        df = (self.n_classes - 1) * self.basis[j].shape[1]
        return float(chi2.sf(max(statistic, 0.0), df))


CI_TESTS = {
    "fisher_z": FisherZ,
    "logistic": LogisticLR,
    "spline": lambda X, y: LogisticLR(X, y, spline=True),
}


def ci_test(name, X, y):
    """A conditional independence test fitted to a table.

    Args:
        name (str): A key of CI_TESTS.
        X (np.ndarray): The features.
        y (np.ndarray): The class labels.

    Returns:
        object: The test, with a `pvalue(j, S)` method.

    Raises:
        ValueError: If the test is not implemented.
    """
    if name not in CI_TESTS:
        raise ValueError(f"Test {name} not implemented ! Available: {list(CI_TESTS)}")
    return CI_TESTS[name](X, y)
