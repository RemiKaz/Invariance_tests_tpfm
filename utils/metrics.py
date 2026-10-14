"""How related a model's perturbed embeddings are to its unperturbed (baseline) ones.

Every metric here takes `(embeds, baseline)`, two (n_test, dim) arrays aligned row by row
(row i is the same test row in both), and returns one float, higher meaning more related.
They differ in what they need that alignment *for*:

- `cosine` and `ccos`: per-row, deterministic measures - row i's value only ever needs row i's
  two vectors, then the per-row values are averaged over the n_test rows.
- `nmi` and `dcka`: neither is defined for a single pair of vectors. Both instead treat the
  n_test rows as a sample from a joint distribution over (baseline, perturbed) and estimate a
  single number for the whole batch.

The raw `mi` and the biased `cka` were dropped: `nmi` and `dcka` are their versions whose scale
does not depend on the dataset (see their docstrings). METRICS maps the --metric CLI choices to
these functions and to the label of their plot axis.
"""

from collections import namedtuple

import numpy as np
from sklearn.feature_selection import mutual_info_regression


def _as_rows(embeds, baseline, name):
    """Both embeddings as (n_test, dim) float arrays, refusing shapes the sample-based
    metrics would silently get wrong (e.g. a leading ensemble axis read as one sample)."""
    embeds, baseline = np.asarray(embeds, dtype=np.float64), np.asarray(baseline, dtype=np.float64)
    if embeds.shape != baseline.shape or embeds.ndim != 2:
        raise ValueError(
            f"{name} needs two (n_test, dim) embeddings of the same shape, got "
            f"{embeds.shape} and {baseline.shape}"
        )
    if len(embeds) < 4:
        raise ValueError(f"{name} needs at least 4 test rows as its sample, got {len(embeds)}")
    return embeds, baseline


def cosine_metric(embeds, baseline):
    """Cosine similarity between each perturbed row and its baseline row, averaged over rows.

    Args:
        embeds (np.ndarray): The perturbed embeddings, shape (n_test, dim).
        baseline (np.ndarray): The unperturbed embeddings, same shape as `embeds`.

    Returns:
        float: The mean cosine similarity.
    """
    embeds, baseline = np.asarray(embeds), np.asarray(baseline)
    num = np.sum(embeds * baseline, axis=-1)
    denom = np.linalg.norm(embeds, axis=-1) * np.linalg.norm(baseline, axis=-1)
    return float((num / denom).mean())


def ccos_metric(embeds, baseline):
    """Centered cosine: cosine similarity after subtracting the baseline's mean embedding
    from both, per test row, then averaged.

    Transformer layers often give every sample a large shared component (the same mean
    vector for all rows). Plain cosine is dominated by it: a layer with a big shared mean
    reads as nearly unchanged whatever happens to the part that differs between samples,
    so comparing cosine across layers can invert their order (a layer with a shared mean
    and more damage read as more invariant than a centered one with less, in every
    synthetic trial). Removing the baseline mean, the same reference for both embeddings,
    measures how each sample's own position moved, which is what differs between layers.
    """
    embeds, baseline = _as_rows(embeds, baseline, "ccos")
    mean = baseline.mean(axis=0)
    return cosine_metric(embeds - mean, baseline - mean)


# Embedding dimensions are typically in the dozens to hundreds; nmi_metric fits one k-NN
# regressor per dimension, so its cost is O(dim) forward passes through sklearn's
# estimator. Past this many dimensions, a fixed, seeded random subset is used instead of
# all of them, to keep a run tractable.
MI_MAX_DIMS = 64


def nmi_metric(embeds, baseline, max_dims=MI_MAX_DIMS, seed=0):
    """Normalized mutual information between baseline and perturbed embeddings, per dimension.

    A single pair of embedding vectors has no defined MI: it is a property of a joint
    distribution, and estimating one needs a sample, here the n_test test rows. So MI is
    estimated one embedding dimension at a time - `baseline[:, d]` against `embeds[:, d]`, with
    the k-nearest-neighbour (Kraskov-Stogbauer-Grassberger) estimator of
    `sklearn.feature_selection.mutual_info_regression` - and divided by the baseline's MI with
    itself, the fraction of each dimension's information the perturbed embedding still carries,
    1 when unchanged. This treats the dimensions as independent: a joint k-NN estimate in the
    full embedding dimension is not reliable at these sample sizes (`dcka` is the joint view).

    The estimator cannot return an infinite MI for a variable against itself: it tops out near
    log(n_test), so raw MI values have a ceiling set by the test-set size (about 2.6 nats on iris
    against 3.6 on digits), not by the embeddings. Dividing each dimension by that ceiling,
    measured on the baseline itself, removes it, so curves can be compared across datasets.
    Dimensions constant over the test set carry no information and are left out.

    Args:
        embeds (np.ndarray): The perturbed embeddings, shape (n_test, dim).
        baseline (np.ndarray): The unperturbed embeddings, same shape as `embeds`.
        max_dims (int, optional): Past this many dimensions, a fixed, seeded random subset is
            used, since the cost scales with the number of dimensions. Defaults to MI_MAX_DIMS.
        seed (int, optional): Seed of that subset, and of the estimator's own tie-breaking
            noise, so repeated calls on the same embeddings agree. Defaults to 0.

    Returns:
        float: The per-dimension normalized MI, averaged over the informative dimensions.
    """
    embeds, baseline = _as_rows(embeds, baseline, "nmi")
    dim = baseline.shape[1]
    dims = np.arange(dim)
    if dim > max_dims:
        dims = np.random.default_rng(seed).choice(dim, size=max_dims, replace=False)

    ratios = []
    for d in dims:
        ceiling = _self_mi(baseline[:, d], seed)
        if ceiling <= 0:
            continue
        mi = mutual_info_regression(baseline[:, [d]], embeds[:, d], random_state=seed)[0]
        ratios.append(mi / ceiling)
    return float(np.mean(ratios)) if ratios else 0.0


# The baseline is the same at every level of an experiment, so its self-MI is cached.
_SELF_MI_CACHE = {}


def _self_mi(column, seed):
    key = (column.tobytes(), seed)
    if key not in _SELF_MI_CACHE:
        if len(_SELF_MI_CACHE) > 100_000:
            _SELF_MI_CACHE.clear()
        _SELF_MI_CACHE[key] = mutual_info_regression(column[:, None], column, random_state=seed)[0]
    return _SELF_MI_CACHE[key]


def _rbf_gram(X):
    """RBF Gram matrix of X, bandwidth set by the median heuristic (median pairwise
    distance of X itself), so it adapts to each embedding's own scale."""
    sq_norms = np.sum(X**2, axis=1)
    # float error can make a same-point distance slightly < 0
    sq_dists = np.clip(sq_norms[:, None] + sq_norms[None, :] - 2 * X @ X.T, 0, None)
    off_diag = sq_dists[~np.eye(len(X), dtype=bool)]
    sigma_sq = max(np.median(off_diag), 1e-12) / 2
    return np.exp(-sq_dists / (2 * sigma_sq))


def _hsic_unbiased(K, L):
    """Unbiased HSIC estimator (Song et al. 2012) of two Gram matrices K, L."""
    n = K.shape[0]
    K, L = K.copy(), L.copy()
    np.fill_diagonal(K, 0)
    np.fill_diagonal(L, 0)
    ones = np.ones(n)
    Lsum_rows = L @ ones
    term = np.sum(K * L) + (K.sum() * L.sum()) / ((n - 1) * (n - 2)) - 2 / (n - 2) * (ones @ K @ Lsum_rows)
    return float(term / (n * (n - 3)))


def dcka_metric(embeds, baseline):
    """Debiased Centered Kernel Alignment between baseline and perturbed embeddings.

    CKA (Kornblith et al. 2019) is built on HSIC, a kernel dependence measure between the
    n_test x n_test Gram matrices (RBF kernel, its bandwidth set from each embedding's own
    median pairwise distance) of the two embeddings: it reads every dimension jointly and never
    estimates a density. The usual, biased HSIC estimator reads two completely independent
    embeddings as similar when the embedding is wide against the number of test rows
    (independent Gaussian embeddings: 0.78 at 50 rows x 64 dims, 0.99 at 188 rows x 4096 dims),
    so its floor changes from one dataset and layer to the next. The unbiased estimator used
    here removes that: independent embeddings read about 0 at any size, identical ones exactly 1.
    It can come out slightly negative (estimation noise around 0). This is the "debiased CKA"
    of Kornblith et al.'s own code (Nguyen et al. 2021).

    Args:
        embeds (np.ndarray): The perturbed embeddings, shape (n_test, dim).
        baseline (np.ndarray): The unperturbed embeddings, same shape as `embeds`.

    Returns:
        float: The debiased CKA similarity, about 0 for independent, 1 for identical.
    """
    embeds, baseline = _as_rows(embeds, baseline, "dcka")
    K, L = _rbf_gram(baseline), _rbf_gram(embeds)
    hsic_xy, hsic_xx, hsic_yy = _hsic_unbiased(K, L), _hsic_unbiased(K, K), _hsic_unbiased(L, L)
    if hsic_xx <= 0 or hsic_yy <= 0:
        return 0.0
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)


def hsic_metric(embeds, baseline):
    """HSIC between baseline and perturbed embeddings, relative to the baseline's own HSIC.

    The same unbiased HSIC estimator and RBF kernels as `dcka_metric`, but normalised by the
    baseline alone, HSIC(base, pert) / HSIC(base, base), where CKA divides by
    sqrt(HSIC(base, base) * HSIC(pert, pert)). It is 1 when the embeddings are unchanged and
    about 0 when they are independent, and unlike CKA it also drops when a perturbation makes the
    perturbed embedding's own structure weaker (less dependence among its rows), not only when
    it rotates it away from the baseline's. Raw HSIC has the scale of the kernels; dividing by a
    quantity of the baseline only keeps every comparison against one baseline (every column of
    a table) on the same scale as raw HSIC, so rankings and ratio-based cuts are unchanged by it.

    Args:
        embeds (np.ndarray): The perturbed embeddings, shape (n_test, dim).
        baseline (np.ndarray): The unperturbed embeddings, same shape as `embeds`.

    Returns:
        float: The relative HSIC, about 0 for independent, 1 for identical.
    """
    embeds, baseline = _as_rows(embeds, baseline, "hsic")
    K, L = _rbf_gram(baseline), _rbf_gram(embeds)
    hsic_xx = _hsic_unbiased(K, K)
    if hsic_xx <= 0:
        return 0.0
    return _hsic_unbiased(K, L) / hsic_xx


Metric = namedtuple("Metric", "fn label")

# The --metric CLI choices: a (embeds, baseline) -> float callable and the label of its axis.
METRICS = {
    "cosine": Metric(cosine_metric, "Cosine similarity to baseline"),
    "ccos": Metric(ccos_metric, "Centered cosine similarity to baseline"),
    "nmi": Metric(nmi_metric, "Normalized mutual information with baseline"),
    "dcka": Metric(dcka_metric, "Debiased CKA similarity to baseline"),
    "hsic": Metric(hsic_metric, "HSIC relative to the baseline's own"),
}


def metric_label(metric):
    """The axis label of a metric, its name for a metric no longer computed."""
    return METRICS[metric].label if metric in METRICS else metric
