"""Linear Gaussian causal models, whose joint distribution is known exactly.

The tables of dataset_generator.py use non-linear mechanisms, for which two different graphs
essentially never give the same distribution. Here every variable is a linear function of its
parents plus independent Gaussian noise,

    X_i = sum_j B[i, j] X_j + eps_i,      eps_i ~ N(0, psi_i)

so the covariance matrix `sigma` of all the variables is the whole distribution, and one can
build graphs with a prescribed relation to each other and check it by arithmetic: the same
`sigma` under another graph (a Markov equivalent one, or one that points the other way), the same
conditional law of one variable given all the others under another `sigma`, ...

A graph is a list of sorted parent lists, one per variable; variables are 0..n-1 and a graph is
acyclic but not necessarily numbered in topological order. Only numpy and scipy are needed.
"""

import numpy as np
from scipy.stats import norm


# ---- graphs ----

def topological_order(parents):
    """A topological order of a graph (parents first, smallest index first among the ready)."""
    n = len(parents)
    waiting = [len(p) for p in parents]
    children = [[j for j in range(n) if i in parents[j]] for i in range(n)]
    ready, order = [i for i in range(n) if waiting[i] == 0], []
    while ready:
        i = min(ready)
        ready.remove(i)
        order.append(i)
        for child in children[i]:
            waiting[child] -= 1
            if waiting[child] == 0:
                ready.append(child)
    if len(order) != n:
        raise ValueError("the graph has a cycle")
    return order


def descendants(parents, node):
    """The set of nodes reachable from `node` by following the edges (it is not included)."""
    children = [[j for j in range(len(parents)) if i in parents[j]] for i in range(len(parents))]
    seen, todo = set(), [node]
    while todo:
        for child in children[todo.pop()]:
            if child not in seen:
                seen.add(child)
                todo.append(child)
    return seen


def skeleton(parents):
    """The set of undirected edges, as frozensets."""
    return {frozenset((p, i)) for i, ps in enumerate(parents) for p in ps}


def v_structures(parents):
    """The set of (frozenset of the two parents, child) where the parents are not adjacent."""
    edges = skeleton(parents)
    return {
        (frozenset((a, b)), i)
        for i, ps in enumerate(parents)
        for a in ps
        for b in ps
        if a < b and frozenset((a, b)) not in edges
    }


def markov_equivalent(parents_a, parents_b):
    """Whether two graphs have the same skeleton and the same v-structures, i.e. the same
    conditional independences, i.e. (for Gaussians) can describe the same distribution."""
    return skeleton(parents_a) == skeleton(parents_b) and v_structures(parents_a) == v_structures(parents_b)


def structural_hamming_distance(parents_a, parents_b):
    """The number of pairs of variables on which two graphs disagree: an edge present in one and
    not the other, or present in both with opposite directions, counts one."""
    n = len(parents_a)
    directed = lambda ps: {(p, i) for i, row in enumerate(ps) for p in row}
    a, b = directed(parents_a), directed(parents_b)
    return sum(
        ((i, j) in a, (j, i) in a) != ((i, j) in b, (j, i) in b)
        for i in range(n)
        for j in range(i + 1, n)
    )


def equivalence_class(parents, limit=300):
    """The graphs of the Markov equivalence class of `parents` (itself included), found by
    reversing covered edges, the moves that never leave the class. At most `limit` graphs."""
    n = len(parents)
    key = lambda ps: tuple(tuple(sorted(p)) for p in ps)
    start = key(parents)
    seen, queue = {start}, [start]
    while queue and len(seen) < limit:
        graph = queue.pop(0)
        for v in range(n):
            for u in graph[v]:
                if set(graph[v]) == set(graph[u]) | {u}:  # u -> v is covered
                    new = [list(p) for p in graph]
                    new[v].remove(u)
                    new[u] = sorted(set(new[u]) | {v})
                    if key(new) not in seen:
                        seen.add(key(new))
                        queue.append(key(new))
    return [[list(p) for p in graph] for graph in sorted(seen)]


# ---- parameters and distributions ----

def covariance(parents, B, psi):
    """The covariance matrix of the model B, psi: (I - B)^-1 diag(psi) (I - B)^-T."""
    inverse = np.linalg.inv(np.eye(len(parents)) - B)
    return inverse @ np.diag(psi) @ inverse.T


def fit_dag(sigma, parents):
    """The parameters (B, psi) of a graph that reproduce `sigma` when the graph is an I-map of it
    (its independences hold in `sigma`): each variable is regressed on its parents."""
    n = len(parents)
    B, psi = np.zeros((n, n)), np.zeros(n)
    for i, ps in enumerate(parents):
        if ps:
            coefficients = np.linalg.solve(sigma[np.ix_(ps, ps)], sigma[ps, i])
            B[i, ps] = coefficients
            psi[i] = sigma[i, i] - sigma[i, ps] @ coefficients
        else:
            psi[i] = sigma[i, i]
    return B, psi


def minimal_imap(sigma, order, tol=1e-9):
    """The sparsest graph that can describe `sigma` when the variables are numbered in `order`:
    each variable has as parents the earlier ones whose regression coefficient is not zero. It
    always works, whatever the order, but a bad order gives a dense graph."""
    parents = [[] for _ in range(len(order))]
    for position, i in enumerate(order):
        earlier = list(order[:position])
        if earlier:
            coefficients = np.linalg.solve(sigma[np.ix_(earlier, earlier)], sigma[earlier, i])
            parents[i] = sorted(e for e, c in zip(earlier, coefficients) if abs(c) > tol)
    return parents


def regress_on_others(sigma, target):
    """The law of `target` given all the other variables: (beta, s2) with
    target | X ~ N(beta . X, s2); beta has a 0 at `target`. beta is non-zero exactly on the
    Markov blanket of the target."""
    n = len(sigma)
    others = [i for i in range(n) if i != target]
    coefficients = np.linalg.solve(sigma[np.ix_(others, others)], sigma[others, target])
    beta = np.zeros(n)
    beta[others] = coefficients
    return beta, float(sigma[target, target] - sigma[target, others] @ coefficients)


def class_thresholds(n_classes, std=1.0):
    """The cuts of a N(0, std^2) variable into `n_classes` classes of equal probability."""
    return std * norm.ppf(np.arange(1, n_classes) / n_classes)


def class_probabilities(values, thresholds, s):
    """P(class | mean) for latent = mean + N(0, s^2) cut at `thresholds`: shape (len(values), K)."""
    cuts = np.concatenate([[-np.inf], thresholds, [np.inf]])
    cdf = norm.cdf((cuts[None, :] - np.asarray(values)[:, None]) / s)
    return np.diff(cdf, axis=1)


def gaussian_kl(sigma_p, sigma_q):
    """KL(N(0, sigma_p) || N(0, sigma_q))."""
    d = len(sigma_p)
    return 0.5 * (
        np.trace(np.linalg.solve(sigma_q, sigma_p)) - d
        + np.linalg.slogdet(sigma_q)[1] - np.linalg.slogdet(sigma_p)[1]
    )


# ---- random models ----

def random_parents(rng, n, in_degree=2, in_degree_mode="fixed", n_roots=1):
    """A random graph over nodes 0..n-1 numbered in topological order: the first `n_roots` have
    no parent, every other node has `in_degree` parents among the earlier ones (or between 1 and
    `in_degree` for in_degree_mode "random")."""
    parents = [[] for _ in range(n_roots)]
    for k in range(n_roots, n):
        degree = in_degree if in_degree_mode == "fixed" else int(rng.integers(1, in_degree + 1))
        parents.append(sorted(rng.choice(k, size=min(k, degree), replace=False).tolist()))
    return parents


def random_parameters(rng, parents, share_range=(0.5, 0.9)):
    """Random weights and noise for a graph numbered in topological order, such that every
    variable has variance 1 and a share `r` of it, drawn in `share_range`, comes from its parents.

    Returns:
        tuple: B, psi and the covariance matrix.
    """
    n = len(parents)
    B, psi, sigma = np.zeros((n, n)), np.ones(n), np.eye(n)
    for k, ps in enumerate(parents):
        if not ps:
            continue
        share = rng.uniform(*share_range)
        weights = rng.choice([-1.0, 1.0], size=len(ps)) * rng.uniform(0.5, 1.5, size=len(ps))
        weights *= np.sqrt(share / (weights @ sigma[np.ix_(ps, ps)] @ weights))
        B[k, ps], psi[k] = weights, 1.0 - share
        sigma[k, :k] = sigma[:k, k] = weights @ sigma[np.ix_(ps, range(k))]
    return B, psi, sigma


def sample(rng, parents, B, psi, n_samples):
    """Rows drawn from the model: array (n_samples, n) with the noise that produced it."""
    noise = rng.normal(size=(n_samples, len(parents))) * np.sqrt(psi)
    values = np.zeros_like(noise)
    for i in topological_order(parents):
        values[:, i] = values @ B[i] + noise[:, i]  # columns of the not yet drawn nodes are still 0
    return values, noise
