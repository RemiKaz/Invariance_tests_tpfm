"""Synthetic tables drawn from a prior, in the spirit of TabICL's SCM priors
(https://github.com/soda-inria/tabicl/tree/main/src/tabicl/prior).

A prior configuration describes a structural causal model; a table is one draw from it. Two
structures are available (the graph and its formula are in scm_graph.py):

- `chain`: a root cause, then every node is a noisy non-linear function of the previous one.
- `dag`: a random causal graph, every node having `in_degree` parents among the nodes before
  it. With `target="blanket"` the target is a node with a parent, a child and a spouse (a
  co-parent of a child that is neither its parent nor its child), so that its Markov blanket
  has every role.

The n_features + 1 nodes are the columns of the table: one of them is the target, the others
are the features. In a chain they stay in chain order; in a DAG their order is shuffled, so
that the position of a column says nothing about the position of its node in the graph. The
target is turned into classes by ranks, as in TabICL's reg2cls.

Only numpy is needed. Example:
    config = PriorConfig(n_samples=2000, n_features=8, n_classes=3, seed=0)
    X, y = generate_table(config)
    generate_and_store(config)      # writes <name>.npz and <name>.json in the data folder

Stored tables are loaded back, split in train and test, by
`datasets.custom_scm_dataset.CustomSCMDataset.load(name)`. From the command line:
    python -m dataset_generator.dataset_generator --n_features 8 --n_classes 3 --seeds 0 1 2
    python -m dataset_generator.dataset_generator --structure dag --in_degree 2 --plot_graph
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone

import numpy as np

from dataset_generator.scm_graph import ACTIVATIONS, has_pure_spouse, markov_blanket, node_info, node_roles, node_value
from datasets.custom_scm_dataset import custom_scm_data_dir, custom_scm_paths

GENERATOR_VERSION = 2
MIN_OUTSIDE = 2  # pool tables: nodes that must stay outside the Markov blanket


@dataclass
class PriorConfig:
    """The prior of a causal chain or DAG table.

    Attributes:
        n_samples (int): The number of rows.
        n_features (int): The number of feature columns; the graph has one more node, the target.
        n_classes (int): The number of classes the target node is cut into.
        target (str | int): Which node is the target: "last" (the end of the chain), "first" (the
            root cause), the index of a node in 0..n_features, or, for a DAG only, "blanket"
            (a random node that has a parent, a child and a spouse that is neither its parent nor its
            child).
        activation (str): The function of every node, a key of ACTIVATIONS, or "random" to draw
            one per node.
        weight_std (float): The scale of the edge weights and biases. In a chain the weights are
            normal with this standard deviation; in a DAG they have a random sign and a
            magnitude between 0.5 and 1.5 times it, so that no edge is (nearly) dead.
        noise_std (float): The noise added at every node but the root, see noise_mode.
        noise_mode (str): "absolute": the noise has standard deviation noise_std, whatever the
            node (a node whose signal is weak is then mostly noise). "relative": it is
            noise_std times the standard deviation of the node's own noiseless value, so every
            node has the same noise power relative to its signal.
        root_sampling (str): The distribution of the root cause, "normal" or "uniform".
        balanced (bool): Equal-sized classes, else the class sizes are drawn from a Dirichlet.
        seed (int | None): The seed of the draw.
        structure (str): "chain" or "dag".
        in_degree (int): DAG only, the number of parents of a node (node k has min(k, in_degree)).
        permute_columns (bool | None): Whether the feature columns are shuffled, hiding the
            order of the nodes. None: True for a DAG, False for a chain.
        n_roots (int): DAG only, the number of nodes without parents (nodes 0..n_roots-1), each
            drawn from the root distribution.
        in_degree_mode (str): DAG only. "fixed": every other node has `in_degree` parents (fewer
            if there are not enough nodes before it). "random": between 1 and `in_degree`, drawn
            for each node.
    """

    n_samples: int = 1000
    n_features: int = 10
    n_classes: int = 2
    target: "str | int" = "last"
    activation: str = "tanh"
    weight_std: float = 1.0
    noise_std: float = 0.1
    noise_mode: str = "absolute"
    root_sampling: str = "normal"
    balanced: bool = True
    seed: "int | None" = None
    structure: str = "chain"
    in_degree: int = 2
    permute_columns: "bool | None" = None
    n_roots: int = 1
    in_degree_mode: str = "fixed"


def _permutes(config):
    return config.structure == "dag" if config.permute_columns is None else config.permute_columns


def _root(config, rng):
    if config.root_sampling == "normal":
        return rng.normal(size=config.n_samples)
    if config.root_sampling == "uniform":
        return rng.random(config.n_samples)
    raise ValueError(f"Unknown root_sampling {config.root_sampling!r}: normal or uniform")


def _check(config):
    if config.structure not in ("chain", "dag"):
        raise ValueError(f"Unknown structure {config.structure!r}: chain or dag")
    if config.noise_mode not in ("absolute", "relative"):
        raise ValueError(f"Unknown noise_mode {config.noise_mode!r}: absolute or relative")
    if config.in_degree_mode not in ("fixed", "random"):
        raise ValueError(f"Unknown in_degree_mode {config.in_degree_mode!r}: fixed or random")
    if config.structure == "dag" and not 1 <= config.n_roots <= config.n_features:
        raise ValueError(f"n_roots must be between 1 and n_features ({config.n_features})")
    if config.activation != "random" and config.activation not in ACTIVATIONS:
        raise ValueError(f"Unknown activation {config.activation!r}: {list(ACTIVATIONS) + ['random']}")


def _add_node(config, rng, nodes, noise, graph, k, activation, weights, bias):
    """Set the parameters of node `k` in the graph and draw its value and its noise."""
    graph["activation"][k], graph["weights"][k], graph["bias"][k] = activation, weights, bias
    signal = node_value(nodes, k, graph)
    scale = config.noise_std * (signal.std() if config.noise_mode == "relative" else 1.0)
    noise[:, k] = scale * rng.normal(size=config.n_samples)
    nodes[:, k] = signal + noise[:, k]


def _empty_graph(parents):
    n = len(parents)
    return {"parents": parents, "activation": [None] * n, "weights": [None] * n, "bias": [None] * n}


def _sample_chain(config, rng):
    """The chain: its nodes and the noise added at each of them (both of shape (n_samples,
    n_features + 1)), and its graph."""
    n = config.n_features + 1
    nodes, noise = np.zeros((config.n_samples, n)), np.zeros((config.n_samples, n))
    nodes[:, 0] = _root(config, rng)
    graph = _empty_graph([[]] + [[k - 1] for k in range(1, n)])
    names = list(ACTIVATIONS)
    for k in range(1, n):
        name = rng.choice(names) if config.activation == "random" else config.activation
        weight, bias = rng.normal(scale=config.weight_std, size=2)
        _add_node(config, rng, nodes, noise, graph, k, str(name), [weight], bias)
    return nodes, noise, graph


def _sample_dag(config, rng):
    """The DAG: nodes, noise and graph like `_sample_chain`, and the target when it is asked to
    be a node with a full Markov blanket (else None)."""
    n = config.n_features + 1
    wants_blanket = config.target == "blanket"
    for _ in range(200):
        parents = [[] for _ in range(config.n_roots)]
        for k in range(config.n_roots, n):
            degree = config.in_degree
            if config.in_degree_mode == "random":
                degree = int(rng.integers(1, config.in_degree + 1))
            parents.append(sorted(rng.choice(k, size=min(k, degree), replace=False).tolist()))
        if not wants_blanket:
            break
        candidates = [k for k in range(n) if parents[k] and has_pure_spouse(parents, k)]
        if candidates:
            break
    else:
        raise ValueError(
            f"No node has a parent, a child and a spouse of its own in {n} nodes with in_degree "
            f"{config.in_degree}: use more features or another in_degree"
        )
    target = int(rng.choice(candidates)) if wants_blanket else None

    nodes, noise = np.zeros((config.n_samples, n)), np.zeros((config.n_samples, n))
    for root in range(config.n_roots):
        nodes[:, root] = _root(config, rng)
    graph = _empty_graph(parents)
    names = list(ACTIVATIONS)
    for k in range(config.n_roots, n):
        name = rng.choice(names) if config.activation == "random" else config.activation
        signs = rng.choice([-1.0, 1.0], size=len(parents[k]))
        weights = (signs * rng.uniform(0.5, 1.5, size=len(parents[k])) * config.weight_std).tolist()
        bias = float(rng.normal(scale=0.5 * config.weight_std))
        _add_node(config, rng, nodes, noise, graph, k, str(name), weights, bias)
    return nodes, noise, graph, target


def _to_classes(values, n_classes, balanced, rng):
    """Cut a continuous target into classes by rank: equal-frequency, or random sizes."""
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    ranks = np.argsort(np.argsort(values)) / len(values)  # spread over [0, 1)
    proportions = np.ones(n_classes) if balanced else rng.dirichlet(np.ones(n_classes))
    cuts = np.cumsum(proportions / proportions.sum())[:-1]
    y = np.searchsorted(cuts, ranks, side="right")
    return np.unique(y, return_inverse=True)[1].astype(np.int64)  # no gap if a class is empty


def _draw(config):
    """One draw: (X, y, nodes, noise, metadata), the seed being resolved when it is None."""
    _check(config)
    if config.seed is None:
        config = replace(config, seed=int(np.random.SeedSequence().entropy % 2**32))
    rng = np.random.default_rng(config.seed)
    if config.structure == "dag":
        nodes, noise, graph, target = _sample_dag(config, rng)
    else:
        nodes, noise, graph = _sample_chain(config, rng)
        target = None
        if config.target == "blanket":
            raise ValueError("target='blanket' needs structure='dag': a chain has no spouse")

    if target is None:
        target = {"last": config.n_features, "first": 0}.get(config.target, config.target)
        if not isinstance(target, (int, np.integer)) or not 0 <= target <= config.n_features:
            raise ValueError(
                f"target must be 'last', 'first', 'blanket' or a node index in 0..{config.n_features}"
            )
    target = int(target)

    y = _to_classes(nodes[:, target], config.n_classes, config.balanced, rng)
    feature_nodes = [k for k in range(nodes.shape[1]) if k != target]
    if _permutes(config):
        feature_nodes = rng.permutation(feature_nodes).tolist()
    X = nodes[:, feature_nodes]

    # class k is target <= thresholds[k] (thresholds ascending): lets a changed target node be
    # turned into labels again with the same cuts
    values = nodes[:, target]
    thresholds = [
        float((values[y == k].max() + values[y == k + 1].min()) / 2) for k in range(y.max())
    ]
    parents = graph["parents"]
    metadata = {
        "generator_version": GENERATOR_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "structure": "causal_dag" if config.structure == "dag" else "causal_chain",
        "config": asdict(config),
        "n_samples": config.n_samples,
        "n_nodes": nodes.shape[1],
        "target_node": target,
        "feature_nodes": feature_nodes,
        "markov_blanket": markov_blanket(parents, target),
        "node_roles": node_roles(parents, target),
        "node_info": node_info(parents),
        "target_thresholds": thresholds,
        "class_counts": np.bincount(y).tolist(),
        "table_arrays": {
            "X": "features, columns are feature_nodes",
            "y": "integer labels of the target node",
            "nodes": "every node, columns are node indices (a topological order)",
            "noise": "noise added at every node (0 for the root)",
        },
    }
    if config.structure == "dag":
        metadata["parents"] = parents
        metadata["node_params"] = [None] * config.n_roots + [
            {"activation": graph["activation"][k], "weights": graph["weights"][k], "bias": graph["bias"][k]}
            for k in range(config.n_roots, nodes.shape[1])
        ]
        metadata["edges"] = [
            {"parent": p, "child": k, "weight": w}
            for k in range(config.n_roots, nodes.shape[1])
            for p, w in zip(parents[k], graph["weights"][k])
        ]
    else:
        metadata["edges"] = [
            {
                "parent": k - 1,
                "child": k,
                "activation": graph["activation"][k],
                "weight": graph["weights"][k][0],
                "bias": graph["bias"][k],
            }
            for k in range(1, nodes.shape[1])
        ]
    return X, y, nodes, noise, metadata


def generate_table(config):
    """Draw one table from the prior.

    Args:
        config (PriorConfig): The prior.

    Returns:
        tuple: `X` of shape (n_samples, n_features), the nodes but the target, and the integer
            labels `y` of shape (n_samples,).
    """
    X, y, _, _, _ = _draw(config)
    return X, y


def default_name(config):
    """A name from the settings of the prior that vary: chain_8f_3c_last_seed0, or
    chain_8f_3c_last_relnoise_seed0 with the relative noise; dag_8f_3c_blanket_deg2_relnoise_seed0
    for a DAG (with `_ordered` when its columns are not shuffled)."""
    noise = "_relnoise" if config.noise_mode == "relative" else ""
    base = f"{config.n_features}f_{config.n_classes}c_{config.target}"
    if config.structure == "dag":
        ordered = "" if _permutes(config) else "_ordered"
        return f"dag_{base}_deg{config.in_degree}{noise}{ordered}_seed{config.seed}"
    return f"chain_{base}{noise}_seed{config.seed}"


def _to_python(value):
    """json default: numpy scalars and arrays as plain python."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not json serialisable")


def store_table(name, X, y, nodes, noise, metadata, data_dir=None):
    """Write one table and its metadata to the data folder: `<name>.npz` (X, y, nodes, noise)
    and `<name>.json`.

    Args:
        name (str): The name of the table.
        X (np.ndarray): The features.
        y (np.ndarray): The integer labels.
        nodes (np.ndarray): Every node, columns in node order.
        noise (np.ndarray): The noise of every node, same shape as `nodes`.
        metadata (dict): The content of the json file; "name" is added to it.
        data_dir (str, optional): The folder. Defaults to `custom_scm_data_dir()`.

    Raises:
        FileExistsError: If a table of that name is already stored.
    """
    data_dir = data_dir or custom_scm_data_dir()
    table_path, metadata_path = custom_scm_paths(name, data_dir)
    if os.path.exists(table_path) or os.path.exists(metadata_path):
        raise FileExistsError(f"A table named {name!r} already exists in {data_dir}")

    os.makedirs(data_dir, exist_ok=True)
    np.savez(table_path, X=X, y=y, nodes=nodes, noise=noise)
    with open(metadata_path, "w") as file:
        json.dump({"name": name, **metadata}, file, indent=2, default=_to_python)
    print(f"Stored {name} in {data_dir}: X {X.shape}, classes {metadata['class_counts']}")


def generate_and_store(config, name=None, data_dir=None):
    """Draw one table from the prior and write it, with all its metadata, to the data folder.

    Two files are written: `<name>.npz` (X, y, nodes, noise) and `<name>.json` (the prior, the
    seed actually used, the graph, which node is the target and which node each column is, its
    Markov blanket and the class thresholds, ...).

    Args:
        config (PriorConfig): The prior. A None seed is drawn and recorded in the metadata.
        name (str, optional): The name of the table. Defaults to `default_name(config)` (which
            needs a seed).
        data_dir (str, optional): The folder. Defaults to `custom_scm_data_dir()`
            (/lustre/.../custom_scm_data, or $CUSTOM_SCM_DATA).

    Returns:
        str: The name of the table.

    Raises:
        FileExistsError: If a table of that name is already stored.
    """
    X, y, nodes, noise, metadata = _draw(config)
    name = name or default_name(replace(config, seed=metadata["config"]["seed"]))
    store_table(name, X, y, nodes, noise, metadata, data_dir)
    return name


def main():
    parser = argparse.ArgumentParser(description="Generate causal chain or DAG tables and store them.")
    defaults = PriorConfig()
    parser.add_argument("--structure", choices=["chain", "dag"], default=defaults.structure)
    parser.add_argument("--in_degree", type=int, default=defaults.in_degree, help="DAG: parents per node")
    parser.add_argument(
        "--no_permute_columns",
        action="store_true",
        help="DAG: keep the feature columns in node order (else they are shuffled)",
    )
    parser.add_argument("--n_samples", type=int, default=defaults.n_samples)
    parser.add_argument("--n_features", type=int, default=defaults.n_features)
    parser.add_argument("--n_classes", type=int, default=defaults.n_classes)
    parser.add_argument(
        "--target",
        default=None,
        help="last, first, a node index or, for a DAG, blanket (default: last for a chain, "
        "blanket for a DAG)",
    )
    parser.add_argument("--activation", default=defaults.activation)
    parser.add_argument("--weight_std", type=float, default=defaults.weight_std)
    parser.add_argument("--noise_std", type=float, default=defaults.noise_std)
    parser.add_argument(
        "--noise_mode",
        choices=["absolute", "relative"],
        default=defaults.noise_mode,
        help="relative: the noise is a fraction (noise_std) of each node's own signal",
    )
    parser.add_argument("--root_sampling", default=defaults.root_sampling)
    parser.add_argument("--unbalanced", action="store_true", help="random class sizes")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="one table per seed")
    parser.add_argument("--name", help="table name, only with a single seed")
    parser.add_argument(
        "--skip_existing", action="store_true", help="leave a table that is already stored alone"
    )
    parser.add_argument(
        "--plot_graph", action="store_true", help="also draw the graph of each stored table"
    )
    parser.add_argument(
        "--names_only", action="store_true", help="only print the table names, nothing is generated"
    )
    parser.add_argument("--data_dir", help="default: $CUSTOM_SCM_DATA or the lustre folder")
    parser.add_argument(
        "--pool",
        type=int,
        default=0,
        help="instead of one configuration: N diverse DAG tables named dagpool_00.., each with "
        "its own configuration drawn from prior_pool.py (the other options are then ignored)",
    )
    args = parser.parse_args()
    if args.name and len(args.seeds) > 1:
        parser.error("--name needs a single seed")
    if args.pool:
        return generate_pool(args)

    target = args.target or ("blanket" if args.structure == "dag" else "last")
    target = int(target) if str(target).isdigit() else target
    for seed in args.seeds:
        config = PriorConfig(
            n_samples=args.n_samples,
            n_features=args.n_features,
            n_classes=args.n_classes,
            target=target,
            activation=args.activation,
            weight_std=args.weight_std,
            noise_std=args.noise_std,
            noise_mode=args.noise_mode,
            root_sampling=args.root_sampling,
            balanced=not args.unbalanced,
            seed=seed,
            structure=args.structure,
            in_degree=args.in_degree,
            permute_columns=False if args.no_permute_columns else None,
        )
        name = args.name or default_name(config)
        if args.names_only:
            print(name)
            continue
        if args.skip_existing and os.path.exists(custom_scm_paths(name, args.data_dir)[0]):
            print(f"Skipping {name}, already stored")
            continue
        generate_and_store(config, name=name, data_dir=args.data_dir)
        if args.plot_graph:
            from dataset_generator.plot_graph import plot_graph

            plot_graph(name, data_dir=args.data_dir)


def generate_pool(args):
    """The `--pool` mode of the command line: N tables of dataset_generator/prior_pool.py."""
    from dataset_generator.prior_pool import pool_name, sample_pool_config

    for index in range(args.pool):
        name = pool_name(index)
        if args.names_only:
            print(name)
            continue
        if args.skip_existing and os.path.exists(custom_scm_paths(name, args.data_dir)[0]):
            print(f"Skipping {name}, already stored")
            continue
        for attempt in range(20):
            config = sample_pool_config(index, attempt=attempt)
            try:
                # the blanket has to leave room for a comparison: at least 2 nodes outside it
                outside = _draw(config)[4]["node_roles"].count("outside")
                if outside < MIN_OUTSIDE:
                    print(f"{name}: attempt {attempt} rejected (only {outside} node(s) outside the blanket)")
                    continue
                generate_and_store(config, name=name, data_dir=args.data_dir)
                break
            except ValueError as error:  # no node with a full blanket in that configuration
                print(f"{name}: attempt {attempt} rejected ({error})")
        else:
            raise RuntimeError(f"No usable configuration found for {name}")
        if args.plot_graph:
            from dataset_generator.plot_graph import plot_graph

            plot_graph(name, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
