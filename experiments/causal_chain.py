import numpy as np
from sklearn.model_selection import train_test_split

from dataset_generator.scm_graph import node_value, resimulate
from datasets.custom_scm_dataset import CustomSCMDataset
from experiments.base_experiment import Experiment


class ChainExperiment(Experiment):
    """A perturbation of the causal model behind a `CustomSCMDataset`, swept over positions.

    Unlike the other experiments the intensity is fixed (a constructor argument) and the swept
    level is *where* in the causal graph the perturbation is made: the perturbed node's
    descendants are recomputed through the stored graph and noise, then the table (features and
    labels, train and test rows alike) is rebuilt with the original split and column order. The
    target keeps the class cuts it was generated with, so it follows if it is a descendant of
    the perturbed node.

    With `propagate=False` (the `_local` variants, registered in experiment_importer.py) the
    descendants are left as they were: the perturbation changes one column whatever its
    position, which separates the effect of where the node sits in the graph from the effect of
    how many columns follow it. Perturbing the target node then perturbs the labels.
    """

    xlabel = "Perturbed node (0 = root cause)"
    # whether the perturbation recomputes a node from its parents even without propagation
    needs_equations = False

    def __init__(self, propagate=True):
        self.propagate = propagate
        if not propagate:
            self.name = f"{self.name}_local"
            self.title = f"{self.title}, not propagated"

    def levels(self, dataset, n_levels):
        if not isinstance(dataset, CustomSCMDataset):
            raise TypeError(f"{self.name} needs a table with a causal graph (dataset_generator or Causal Chamber)")
        if dataset.noise is None and (self.propagate or self.needs_equations):
            # real measurements (a Causal Chamber table): there is no structural equation to
            # recompute a node from its parents with
            raise TypeError(f"{self.name} needs the structural equations of a simulated table: on "
                            f"{dataset.name}, run the column-only experiments (--exp blanket)")
        return np.arange(self.n_positions(dataset))

    def n_positions(self, dataset):
        return dataset.nodes.shape[1]

    def perturb(self, dataset, level, rng):
        nodes = self.perturb_nodes(dataset, level, rng)
        target = dataset.target_node
        y = np.searchsorted(dataset.metadata["target_thresholds"], nodes[:, target]).astype(np.int64)
        X = nodes[:, dataset.feature_nodes]
        # the split of from_xy depends on the number of rows only, so it can be replayed
        train, test = train_test_split(
            np.arange(len(y)), test_size=dataset.test_size, random_state=dataset.seed
        )
        return dataset.perturbed(X_train=X[train], y_train=y[train], X_test=X[test], y_test=y[test])

    def perturb_nodes(self, dataset, position, rng):
        raise NotImplementedError

    def resimulate(self, nodes, dataset, changed):
        """The nodes after a change made at node `changed`: its descendants recomputed, or left
        as they were when the experiment does not propagate."""
        if not self.propagate:
            return nodes
        return resimulate(nodes, dataset.noise, dataset.graph, [changed])


class ChainNoise(ChainExperiment):
    """Add noise to one node, of standard deviation `strength` times the one of that node.

    Args:
        strength (float, optional): The std of the noise, in stds of the node. Defaults to 1.0.
        kind (str, optional): Its distribution, at the same variance: "gaussian", "uniform"
            (bounded) or "laplace" (heavy tails). Defaults to "gaussian".
        rows (str, optional): The rows noised: "all", "train" (the context only: what the
            model learns from changes, the queries are clean) or "test" (the queries only: the
            fitted function is read at noisy points, classic permutation importance on a fixed
            model). The split is the dataset's own. Defaults to "all".
        propagate (bool, optional): Recompute the descendants. Defaults to True.
    """

    name = "chain_noise"
    title = "Extra noise on one node"
    KINDS = {
        "gaussian": lambda rng, n: rng.normal(size=n),
        "uniform": lambda rng, n: rng.uniform(-np.sqrt(3), np.sqrt(3), size=n),
        "laplace": lambda rng, n: rng.laplace(scale=1 / np.sqrt(2), size=n),
    }

    def __init__(self, strength=1.0, kind="gaussian", rows="all", propagate=True):
        super().__init__(propagate)
        if kind not in self.KINDS or rows not in ("all", "train", "test"):
            raise ValueError(f"kind in {list(self.KINDS)}, rows in all / train / test")
        self.strength, self.kind, self.rows = strength, kind, rows

    def perturb_nodes(self, dataset, position, rng):
        nodes = dataset.nodes.copy()
        column = nodes[:, position]
        noise = self.strength * column.std() * self.KINDS[self.kind](rng, len(column))
        if self.rows != "all":
            # the split of from_xy depends on the number of rows only, so it is replayed
            train, test = train_test_split(np.arange(len(nodes)), test_size=dataset.test_size, random_state=dataset.seed)
            keep = np.zeros(len(nodes), dtype=bool)
            keep[train if self.rows == "train" else test] = True
            noise = np.where(keep, noise, 0.0)
        nodes[:, position] = column + noise
        return self.resimulate(nodes, dataset, position)


class ChainRandomize(ChainExperiment):
    """Randomize one node: its values are permuted across the rows, which keeps its
    distribution but cuts its dependence on its parent (and on the rows of its descendants)."""

    name = "chain_randomize"
    title = "Randomize one node"

    def perturb_nodes(self, dataset, position, rng):
        nodes = dataset.nodes.copy()
        nodes[:, position] = nodes[rng.permutation(len(nodes)), position]
        return self.resimulate(nodes, dataset, position)


class ChainEdgeShuffle(ChainExperiment):
    """Instance-wise perturbation of the edges into one node: for a `fraction` of the rows, the
    node is computed from the parent values of another row (the same row for all its parents),
    its own noise being kept. The nodes keep their distributions, but the link between the node
    and its parents is broken for those instances. Level k is the node k+1; in a chain, the edge
    from node k to node k+1."""

    name = "chain_edge_shuffle"
    title = "Shuffle rows between a node and its parents"
    xlabel = "Perturbed edge k into node k+1"
    needs_equations = True  # the child is recomputed from the shuffled parents

    def __init__(self, fraction=0.5, propagate=True):
        super().__init__(propagate)
        self.fraction = fraction

    def n_positions(self, dataset):
        return dataset.nodes.shape[1] - 1

    def perturb_nodes(self, dataset, position, rng):
        graph, child = dataset.graph, position + 1
        parents = graph["parents"][child]
        nodes = dataset.nodes.copy()
        rows = rng.choice(len(nodes), size=int(self.fraction * len(nodes)), replace=False)
        shuffled = nodes.copy()
        shuffled[np.ix_(rows, parents)] = nodes[np.ix_(rng.permutation(rows), parents)]
        nodes[:, child] = node_value(shuffled, child, graph) + dataset.noise[:, child]
        return self.resimulate(nodes, dataset, child)



class ChainCondRandomize(ChainExperiment):
    """Randomize one node given the others: it is replaced by its prediction from the other
    feature nodes plus its own residuals permuted across the rows (conditional permutation,
    Strobl et al. 2008; a knockoff-like resampling).

    A plain permutation (`chain_randomize`) also cuts the node from the features it is
    correlated with, which makes rows no table of the system would hold (a light sensor reading
    that does not match the light source), and a model may react to that whatever the node's
    role. Here the dependence on the other features is kept: if the node is outside the Markov
    blanket of the target, the perturbed table is (up to the fit of the prediction) a draw of
    the same distribution, so a perfect P(y | X) does not change; only a blanket node changes it.

    The prediction is out-of-fold (5 folds): a ridge regression on a cubic spline of every other
    feature node and their pairwise products, the target node left out (it must not leak into
    the resampling). It does not depend on the seed and is computed once per table and node.
    """

    name = "chain_condrandomize"
    title = "Randomize one node given the others"

    def __init__(self, propagate=True):
        super().__init__(propagate)
        self._predictions = {}  # (table, node) -> out-of-fold prediction of the node

    def prediction(self, dataset, position):
        key = (dataset.name, position, dataset.nodes.shape)
        if key not in self._predictions:
            from sklearn.linear_model import RidgeCV
            from sklearn.model_selection import KFold, cross_val_predict
            from sklearn.preprocessing import SplineTransformer, StandardScaler

            others = [k for k in dataset.feature_nodes if k != position]
            Z = StandardScaler().fit_transform(dataset.nodes[:, others])
            design = [SplineTransformer(n_knots=5, degree=3).fit_transform(Z[:, [i]]) for i in range(len(others))]
            design += [Z[:, [i]] * Z[:, [j]] for i in range(len(others)) for j in range(i + 1, len(others))]
            model = RidgeCV(alphas=np.logspace(-3, 3, 13))
            folds = KFold(5, shuffle=True, random_state=0)
            self._predictions[key] = cross_val_predict(model, np.hstack(design), dataset.nodes[:, position], cv=folds)
        return self._predictions[key]

    def perturb_nodes(self, dataset, position, rng):
        nodes = dataset.nodes.copy()
        predicted = self.prediction(dataset, position)
        residuals = nodes[:, position] - predicted
        nodes[:, position] = predicted + residuals[rng.permutation(len(nodes))]
        return self.resimulate(nodes, dataset, position)
