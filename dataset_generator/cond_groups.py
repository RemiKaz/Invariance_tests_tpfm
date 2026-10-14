"""Groups of tables to ask how a model depends on P(y | X), and not on the causal graph.

A group is five tables built from a baseline, in a linear Gaussian model (see gaussian_scm.py)
where the target is a continuous node T of the graph and the label y is T cut into classes at
fixed thresholds. Everything about P(X, y) and P(y | X) is then known exactly, so each relation
below holds by construction and is checked by arithmetic, and by a calibration test on the sample:

- `baseline`:     a random DAG in which the target has a parent, a child and a spouse.
- `markov_equiv`: the same joint distribution P(X, y), another graph of its Markov equivalence
                  class (edges reversed). Same P(y | X), same P(X); the roles of some nodes
                  towards the target change.
- `anticausal`:   the same joint distribution again, with the graph that has the target first and
                  the features as its descendants (y -> X): the sparsest graph for that order.
- `same_cond`:    the same P(y | X) (same regression of T on X, same thresholds) with another
                  P(X) and another graph: a new random DAG among the features, T a child of the
                  baseline's Markov blanket.
- `near_graph`:   the baseline graph with one edge into the target added (or removed), P(y | X)
                  changed on purpose.

The tables of a group are independent samples, with the same variable in the same column. They are
stored like the others (dataset_generator.py, `CustomSCMDataset.load(name)` reads them), with the
relation to the baseline and its checks in the metadata under "group".

    python -m dataset_generator.cond_groups --n_groups 3 --seed 0
"""

import argparse
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone

import numpy as np

from dataset_generator import gaussian_scm as g
from dataset_generator.dataset_generator import GENERATOR_VERSION, store_table
from dataset_generator.scm_graph import has_pure_spouse, markov_blanket, node_info, node_roles, node_value
from datasets.custom_scm_dataset import custom_scm_paths

VARIANTS = ("baseline", "markov_equiv", "same_cond", "anticausal", "near_graph")
DESCRIPTIONS = {
    "baseline": "the reference table",
    "markov_equiv": "same joint P(X, y), another graph of the Markov equivalence class",
    "same_cond": "same P(y | X), another P(X) and another graph",
    "anticausal": "same joint P(X, y), the graph with the target first (y -> X)",
    "near_graph": "the baseline graph with one edge into the target added or removed, P(y | X) changed",
}
SEED = 2000
MIN_COND_TV = 0.08  # near_graph: the mean total variation distance of P(y | X) asked for
MIN_X_KL, MAX_X_KL = 0.5, 20.0  # same_cond: the KL divergence between the two P(X) asked for (nats)


@dataclass
class GroupConfig:
    """What varies from one group to the next.

    Attributes:
        n_features (int): The number of features; the target is one more variable.
        n_classes (int): The number of classes the target is cut into.
        n_samples (int): The number of rows of each table.
        in_degree (int): The number of parents of a node of the baseline (at most).
        in_degree_mode (str): "fixed" or "random", as in dataset_generator.PriorConfig.
        n_roots (int): The number of nodes of the baseline without parents.
        share_range (tuple): The range of the share of a node's variance that comes from its
            parents (the rest is its own noise).
    """

    n_features: int = 8
    n_classes: int = 3
    n_samples: int = 1000
    in_degree: int = 2
    in_degree_mode: str = "fixed"
    n_roots: int = 1
    share_range: tuple = (0.5, 0.9)


def sample_group_config(index, seed=SEED, n_samples=1000):
    """The configuration of group `index`, always the same for a given seed."""
    rng = np.random.default_rng([seed, index, 99])
    return GroupConfig(
        n_features=int(rng.choice([6, 8, 10])),
        n_classes=int(rng.choice([2, 3])),
        n_samples=n_samples,
        in_degree=int(rng.choice([2, 3])),
        in_degree_mode=str(rng.choice(["fixed", "random"])),
        n_roots=int(rng.choice([1, 2])),
    )


def group_names(index):
    return {variant: f"cg{index:02d}_{variant}" for variant in VARIANTS}


# ---- the models of a group (variables 0..d-1 are the features, d is the target) ----

@dataclass
class Model:
    parents: list
    B: np.ndarray
    psi: np.ndarray

    @property
    def sigma(self):
        return g.covariance(self.parents, self.B, self.psi)


def _relabel(parents, B, psi, var_of_position):
    """The same model with node `k` renamed `var_of_position[k]`."""
    idx = np.asarray(var_of_position)
    n = len(idx)
    new_parents, new_B, new_psi = [None] * n, np.zeros((n, n)), np.zeros(n)
    for k, ps in enumerate(parents):
        new_parents[idx[k]] = sorted(int(idx[p]) for p in ps)
    new_B[np.ix_(idx, idx)] = B
    new_psi[idx] = psi
    return new_parents, new_B, new_psi


def _role_changes(parents_a, parents_b, target):
    roles_a, roles_b = node_roles(parents_a, target), node_roles(parents_b, target)
    return sum(a != b for a, b in zip(roles_a, roles_b))


def build_baseline(config, rng, attempts=3000):
    """A random model whose target has a parent, a child and a spouse, and whose Markov
    equivalence class has a graph in which the role of some node towards the target differs."""
    d = config.n_features
    n = d + 1
    for _ in range(attempts):
        parents = g.random_parents(rng, n, config.in_degree, config.in_degree_mode, config.n_roots)
        candidates = [k for k in range(n) if parents[k] and has_pure_spouse(parents, k)]
        if not candidates:
            continue
        target = int(rng.choice(candidates))
        B, psi, _ = g.random_parameters(rng, parents, config.share_range)
        features = iter(rng.permutation(d).tolist())
        var_of_position = [d if k == target else next(features) for k in range(n)]
        parents_v, B_v, psi_v = _relabel(parents, B, psi, var_of_position)
        if any(_role_changes(parents_v, other, d) for other in g.equivalence_class(parents_v)):
            return Model(parents_v, B_v, psi_v)
    raise RuntimeError("no baseline with a usable Markov equivalence class was found")


def build_markov_equiv(base, rng):
    """Another graph of the baseline's Markov equivalence class, refit on the same covariance,
    the one that changes the most roles towards the target."""
    d = len(base.parents) - 1
    others = [graph for graph in g.equivalence_class(base.parents) if graph != base.parents]
    changes = [_role_changes(base.parents, graph, d) for graph in others]
    best = [graph for graph, c in zip(others, changes) if c == max(changes)]
    parents = best[int(rng.integers(len(best)))]
    B, psi = g.fit_dag(base.sigma, parents)
    return Model(parents, B, psi)


def build_anticausal(base):
    """The same distribution with the variables ordered target first, then the others in the
    reverse of the baseline's topological order: the target is a root cause of the features."""
    d = len(base.parents) - 1
    order = [d] + [v for v in reversed(g.topological_order(base.parents)) if v != d]
    parents = g.minimal_imap(base.sigma, order)
    B, psi = g.fit_dag(base.sigma, parents)
    return Model(parents, B, psi)


def build_same_cond(base, config, rng, attempts=1500, wanted=40):
    """The baseline's P(y | X) with another P(X): a new random DAG among the features, with unit
    variances like the baseline's, and the target a child of the baseline's Markov blanket with
    the baseline's regression coefficients and residual variance. The new target has a variance
    close to 1, so that the fixed thresholds still give classes of similar sizes. Of `wanted`
    models that fit, the one whose P(X) is closest to the baseline's (but at least MIN_X_KL away)
    is kept, so that P(X) differs without being extreme."""
    d = len(base.parents) - 1
    sigma = base.sigma
    beta, s2 = g.regress_on_others(sigma, d)
    support = [j for j in range(d) if abs(beta[j]) > 1e-9]
    found = []
    for attempt in range(attempts):
        tolerance = 0.1 if attempt < attempts // 2 else 0.25
        parents = g.random_parents(rng, d, config.in_degree, config.in_degree_mode, config.n_roots)
        B, psi, _ = g.random_parameters(rng, parents, config.share_range)
        features = rng.permutation(d).tolist()
        parents, B, psi = _relabel(parents, B, psi, features)
        sigma_x = g.covariance(parents, B, psi)
        variance_t = beta[:d] @ sigma_x @ beta[:d] + s2
        kl = g.gaussian_kl(sigma[:d, :d], sigma_x)
        if abs(variance_t - 1.0) > tolerance or not MIN_X_KL <= kl <= MAX_X_KL:
            continue
        full_B = np.zeros((d + 1, d + 1))
        full_B[:d, :d] = B
        full_B[d, :d] = beta[:d]
        found.append((kl, Model(parents + [support], full_B, np.concatenate([psi, [s2]]))))
        if len(found) == wanted:
            break
    if not found:
        raise RuntimeError("no P(X) with the baseline's P(y | X) and a target variance near 1 was found")
    return min(found, key=lambda item: item[0])[1]


def conditional_tv(base, other, config, rng, n=20000):
    """Mean total variation distance between the two models' P(y | X), over X ~ the baseline."""
    d = len(base.parents) - 1
    sigma0, sigma1 = base.sigma, other.sigma
    X = rng.multivariate_normal(np.zeros(d), sigma0[:d, :d], size=n)
    thresholds = g.class_thresholds(config.n_classes)
    (beta0, s0), (beta1, s1) = g.regress_on_others(sigma0, d), g.regress_on_others(sigma1, d)
    p0 = g.class_probabilities(X @ beta0[:d], thresholds, np.sqrt(s0))
    p1 = g.class_probabilities(X @ beta1[:d], thresholds, np.sqrt(s1))
    return float(0.5 * np.abs(p0 - p1).sum(axis=1).mean())


def build_near_graph(base, config, rng):
    """The baseline with one edge into the target added (from a node that is neither a
    descendant of the target nor already its parent) or removed. The explained share of the
    target's variance is kept, so its variance stays 1. Among the edits, the first (in random
    order, additions before removals) that changes P(y | X) by at least MIN_COND_TV, else the
    one that changes it most."""
    d = len(base.parents) - 1
    sigma = base.sigma
    share = 1.0 - base.psi[d]
    current = list(base.parents[d])
    forbidden = set(g.descendants(base.parents, d)) | set(current) | {d}
    additions = [("add", u) for u in range(d) if u not in forbidden]
    removals = [("remove", p) for p in current]
    edits = [additions[i] for i in rng.permutation(len(additions))] + [removals[i] for i in rng.permutation(len(removals))]
    typical = float(np.mean(np.abs(base.B[d, current])))
    best = None
    for kind, node in edits:
        new_parents = sorted(set(current) | {node}) if kind == "add" else [p for p in current if p != node]
        weights = np.zeros(d + 1)
        weights[current] = base.B[d, current]
        if kind == "add":
            weights[node] = rng.choice([-1.0, 1.0]) * typical
        else:
            weights[node] = 0.0
        if new_parents:
            weights *= np.sqrt(share / (weights[new_parents] @ sigma[np.ix_(new_parents, new_parents)] @ weights[new_parents]))
        B = base.B.copy()
        B[d] = weights if new_parents else 0.0
        psi = base.psi.copy()
        psi[d] = base.psi[d] if new_parents else 1.0
        parents = [list(p) for p in base.parents]
        parents[d] = new_parents
        candidate = Model(parents, B, psi)
        tv = conditional_tv(base, candidate, config, rng)
        if best is None or tv > best[0]:
            best = (tv, candidate)
        if tv >= MIN_COND_TV:
            return candidate
    return best[1]


# ---- tables ----

def _expected_calibration_error(probabilities, labels, bins=10):
    """How far the top-class probabilities are from the observed frequency of being right."""
    top, predicted = probabilities.max(axis=1), probabilities.argmax(axis=1)
    right = (predicted == labels).astype(float)
    edges = np.linspace(top.min(), top.max() + 1e-12, bins + 1)
    which = np.clip(np.digitize(top, edges) - 1, 0, bins - 1)
    return float(sum(abs(right[which == b].mean() - top[which == b].mean()) * (which == b).mean() for b in range(bins) if (which == b).any()))


def _checks(base, model, variant, config, values, rng):
    """The relation of a variant to the baseline, as numbers: exact (population) quantities and
    one measured on the sample (the calibration of the baseline's P(y | X) on this table)."""
    d = len(base.parents) - 1
    sigma0, sigma1 = base.sigma, model.sigma
    (beta0, s0), (beta1, s1) = g.regress_on_others(sigma0, d), g.regress_on_others(sigma1, d)
    thresholds = g.class_thresholds(config.n_classes)
    labels = np.searchsorted(thresholds, values[:, d])
    probabilities = g.class_probabilities(values[:, :d] @ beta0[:d], thresholds, np.sqrt(s0))
    return {
        "structural_hamming_distance": g.structural_hamming_distance(base.parents, model.parents),
        "markov_equivalent_to_baseline": bool(g.markov_equivalent(base.parents, model.parents)),
        "same_skeleton": bool(g.skeleton(base.parents) == g.skeleton(model.parents)),
        "role_changes_towards_target": _role_changes(base.parents, model.parents, d),
        "joint_max_abs_diff": float(np.abs(sigma0 - sigma1).max()),
        "conditional_beta_max_abs_diff": float(np.abs(beta0 - beta1).max()),
        "conditional_residual_variance_diff": float(abs(s0 - s1)),
        "conditional_mean_tv_distance": 0.0 if variant == "baseline" else conditional_tv(base, model, config, rng),
        "x_kl_baseline_to_variant": float(g.gaussian_kl(sigma0[:d, :d], sigma1[:d, :d])),
        "target_variance": float(sigma1[d, d]),
        "calibration_error_of_baseline_posterior_on_sample": _expected_calibration_error(probabilities, labels),
        "n_edges": int(sum(len(p) for p in model.parents)),
    }


def make_table(model, base, variant, name, group_index, config, column_variables, rng, seed):
    """One table of a group: sample, node order, stored mechanisms, metadata.

    The nodes are numbered in a topological order of this model's graph, and the column `j` of X
    is always the variable `column_variables[j]`. The mechanisms are stored as an identity
    activation on the standardized parents, which reproduces the sample exactly (`node_value`
    plus the stored noise gives the nodes back), so that the perturbation experiments can
    propagate through this graph.
    """
    d = config.n_features
    values, noise_by_variable = g.sample(rng, model.parents, model.B, model.psi, config.n_samples)
    order = g.topological_order(model.parents)
    index_of = {v: k for k, v in enumerate(order)}
    nodes, noise = values[:, order], noise_by_variable[:, order]
    n_nodes = len(order)
    parents = [sorted(index_of[p] for p in model.parents[v]) for v in order]
    activation, weights, bias = [None] * n_nodes, [None] * n_nodes, [None] * n_nodes
    for k, v in enumerate(order):
        if not parents[k]:
            noise[:, k] = 0.0  # a root keeps its value in `nodes`, there is nothing to recompute
            continue
        raw = np.array([model.B[v, order[j]] for j in parents[k]])
        scale = np.array([nodes[:, j].std() + 1e-8 for j in parents[k]])
        means = np.array([nodes[:, j].mean() for j in parents[k]])
        activation[k] = "identity"
        weights[k] = (raw * scale * np.sqrt(len(parents[k]))).tolist()
        bias[k] = float(raw @ means)
    graph = {"parents": parents, "activation": activation, "weights": weights, "bias": bias}
    for k in range(n_nodes):
        if parents[k]:
            assert np.allclose(node_value(nodes, k, graph) + noise[:, k], nodes[:, k], atol=1e-6), "stored mechanism"

    thresholds = g.class_thresholds(config.n_classes).tolist()
    target = index_of[d]
    y = np.searchsorted(thresholds, nodes[:, target]).astype(np.int64)
    feature_nodes = [index_of[v] for v in column_variables]
    X = nodes[:, feature_nodes]

    checks = _checks(base, model, variant, config, values, rng)
    metadata = {
        "generator_version": GENERATOR_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "structure": "causal_dag",
        "config": {
            "n_samples": config.n_samples, "n_features": d, "n_classes": config.n_classes, "target": "blanket",
            "activation": "identity", "noise_mode": "gaussian_linear", "family": "linear_gaussian",
            "balanced": False, "seed": seed, "structure": "dag", "in_degree": config.in_degree,
            "in_degree_mode": config.in_degree_mode, "n_roots": config.n_roots, "permute_columns": True,
        },
        "n_samples": config.n_samples,
        "n_nodes": n_nodes,
        "target_node": target,
        "feature_nodes": feature_nodes,
        "markov_blanket": markov_blanket(parents, target),
        "node_roles": node_roles(parents, target),
        "node_info": node_info(parents),
        "target_thresholds": thresholds,
        "class_counts": np.bincount(y, minlength=config.n_classes).tolist(),
        "table_arrays": {
            "X": "features, columns are feature_nodes",
            "y": "integer labels of the target node",
            "nodes": "every node, columns are node indices (a topological order)",
            "noise": "noise added at every node (0 for the root)",
        },
        "parents": parents,
        "node_params": [None if not parents[k] else {"activation": "identity", "weights": weights[k], "bias": bias[k]} for k in range(n_nodes)],
        "edges": [{"parent": j, "child": k, "weight": float(model.B[order[k], order[j]])} for k in range(n_nodes) for j in parents[k]],
        "group": {
            "index": group_index,
            "variant": variant,
            "description": DESCRIPTIONS[variant],
            "baseline": f"cg{group_index:02d}_baseline",
            "variable_of_node": order,
            "column_variables": column_variables,
            "target_variable": d,
            "relation_to_baseline": checks,
        },
    }
    return X, y, nodes, noise, metadata


def generate_group(index, seed=SEED, n_samples=1000, data_dir=None, skip_existing=False):
    """Build and store the five tables of group `index`.

    Returns:
        dict: variant -> the relation checks of its table (see `_checks`).
    """
    config = sample_group_config(index, seed, n_samples)
    rng = np.random.default_rng([seed, index, 0])
    # a dense graph leaves few edges free to reverse: if the sampled configuration has no usable
    # baseline, the next ones are sparser
    fallbacks = [config, replace(config, in_degree=2, in_degree_mode="random"), replace(config, in_degree=2, in_degree_mode="fixed", n_roots=config.n_roots + 1)]
    for candidate in fallbacks:
        try:
            base, config = build_baseline(candidate, rng), candidate
            break
        except RuntimeError:
            print(f"group {index:02d}: no usable baseline with {candidate}, trying a sparser configuration")
    else:
        raise RuntimeError(f"group {index}: no usable baseline")
    models = {
        "baseline": base,
        "markov_equiv": build_markov_equiv(base, rng),
        "same_cond": build_same_cond(base, config, rng),
        "anticausal": build_anticausal(base),
        "near_graph": build_near_graph(base, config, rng),
    }
    column_variables = rng.permutation(config.n_features).tolist()
    names, report = group_names(index), {}
    for k, variant in enumerate(VARIANTS):
        sample_rng = np.random.default_rng([seed, index, 1 + k])
        table_seed = int(sample_rng.integers(0, 2**31 - 1))
        X, y, nodes, noise, metadata = make_table(
            models[variant], base, variant, names[variant], index, config, column_variables, sample_rng, table_seed
        )
        report[variant] = metadata["group"]["relation_to_baseline"]
        if skip_existing and os.path.exists(custom_scm_paths(names[variant], data_dir)[0]):
            print(f"Skipping {names[variant]}, already stored")
            continue
        store_table(names[variant], X, y, nodes, noise, metadata, data_dir)
    return report


def print_report(index, report):
    """The relation of each table of a group to its baseline, one line per table."""
    print(f"\ngroup {index:02d}: relation of each table to the baseline")
    print("variant        SHD  markov-eq  roles changed  |joint diff|  |beta diff|  mean TV of P(y|X)  KL of P(X)  calibration err")
    for variant in VARIANTS:
        r = report[variant]
        print(
            f"{variant:13s} {r['structural_hamming_distance']:4d}  {str(r['markov_equivalent_to_baseline']):9s}  {r['role_changes_towards_target']:13d}  "
            f"{r['joint_max_abs_diff']:11.1e}  {r['conditional_beta_max_abs_diff']:10.1e}  {r['conditional_mean_tv_distance']:17.3f}  "
            f"{r['x_kl_baseline_to_variant']:10.3f}  {r['calibration_error_of_baseline_posterior_on_sample']:15.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_groups", type=int, default=3, help="number of groups, 5 tables each")
    parser.add_argument("--first_group", type=int, default=0, help="index of the first group")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--data_dir", help="default: $CUSTOM_SCM_DATA or the lustre folder")
    parser.add_argument("--skip_existing", action="store_true", help="leave tables that are already stored alone")
    parser.add_argument("--plot_graph", action="store_true", help="also draw the graph of each stored table")
    parser.add_argument("--names_only", action="store_true", help="only print the table names")
    args = parser.parse_args()

    for index in range(args.first_group, args.first_group + args.n_groups):
        if args.names_only:
            print("\n".join(group_names(index).values()))
            continue
        report = generate_group(index, args.seed, args.n_samples, args.data_dir, args.skip_existing)
        print_report(index, report)
        if args.plot_graph:
            from dataset_generator.plot_graph import plot_graph

            for name in group_names(index).values():
                plot_graph(name, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
