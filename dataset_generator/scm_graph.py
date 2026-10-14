"""The causal graph behind a generated table, and what can be done with it.

A table is drawn from a structural causal model whose nodes are numbered in topological order
(every node's parents have a smaller index): a root, then

    node_k = activation_k(sum_j w_kj * standardize(node_j) / sqrt(n_parents) + b_k) + noise_k

over the parents j of k. A chain is the case where node k's only parent is node k-1. The graph
is stored as a dict of lists indexed by node:

    {"parents": [[], [0], [0, 1], ...], "activation": [None, "tanh", ...],
     "weights": [None, [0.7], [-1.1, 0.4], ...], "bias": [None, 0.1, ...]}

This module only needs numpy, it is shared by the generator, the dataset and the experiments.
"""

import numpy as np

ACTIVATIONS = {
    "identity": lambda x: x,
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(x, 0.0),
    "leaky_relu": lambda x: np.where(x > 0, x, 0.1 * x),
    "sin": np.sin,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "abs": np.abs,
}

ROLES = ("target", "parent", "child", "spouse", "outside")


def graph_of(metadata):
    """The graph of a stored table, whether it is a chain (its `edges`) or a DAG.

    A table of real measurements (a Causal Chamber, see dataset_generator/causal_chambers.py)
    has `parents` but no structural equations: its activations, weights and biases are None.

    Args:
        metadata (dict): The content of the table's json file.

    Returns:
        dict: The graph, see the module docstring.
    """
    if "node_params" not in metadata and "parents" in metadata:
        n = len(metadata["parents"])
        return {"parents": [list(p) for p in metadata["parents"]], "activation": [None] * n,
                "weights": [None] * n, "bias": [None] * n}
    if "node_params" in metadata:
        params = metadata["node_params"]
        return {
            "parents": [list(p) for p in metadata["parents"]],
            "activation": [p and p["activation"] for p in params],
            "weights": [p and list(p["weights"]) for p in params],
            "bias": [p and p["bias"] for p in params],
        }
    edges = metadata["edges"]
    n = len(edges) + 1
    graph = {"parents": [[] for _ in range(n)], "activation": [None] * n, "weights": [None] * n, "bias": [None] * n}
    for edge in edges:
        child = edge["child"]
        graph["parents"][child] = [edge["parent"]]
        graph["activation"][child] = edge["activation"]
        graph["weights"][child] = [edge["weight"]]
        graph["bias"][child] = edge["bias"]
    return graph


def node_value(nodes, k, graph):
    """The noiseless value of node `k` from the current values of its parents.

    Args:
        nodes (np.ndarray): The nodes, of shape (n_samples, n_nodes); only the columns of the
            parents of `k` are read.
        k (int): The node.
        graph (dict): The graph.

    Returns:
        np.ndarray: The value of node `k` over the rows.
    """
    parents = graph["parents"][k]
    total = 0.0
    for weight, parent in zip(graph["weights"][k], parents):
        column = nodes[:, parent]
        total = total + weight * ((column - column.mean()) / (column.std() + 1e-8))
    return ACTIVATIONS[graph["activation"][k]](total / np.sqrt(len(parents)) + graph["bias"][k])


def resimulate(nodes, noise, graph, changed):
    """Recompute the descendants of some changed nodes, keeping the stored noise.

    This is what an intervention on a node does to the rest of the graph: everything downstream
    follows the change through the same edges and the same noise. The changed nodes keep the
    values they were given.

    Args:
        nodes (np.ndarray): The nodes, of shape (n_samples, n_nodes).
        noise (np.ndarray): The noise of every node, same shape.
        graph (dict): The graph.
        changed (list): The nodes whose values were changed.

    Returns:
        np.ndarray: A copy of `nodes` with the descendants of `changed` recomputed.
    """
    nodes = nodes.copy()
    dirty = set(changed)
    for k, parents in enumerate(graph["parents"]):
        if k not in changed and dirty.intersection(parents):
            nodes[:, k] = node_value(nodes, k, graph) + noise[:, k]
            dirty.add(k)
    return nodes


def children_of(parents):
    """The children of every node."""
    return [[j for j, ps in enumerate(parents) if i in ps] for i in range(len(parents))]


def markov_blanket(parents, target):
    """The Markov blanket of a node: its parents, its children and its children's other parents.

    Returns:
        dict: "parents", "children" and "spouses", each a sorted list of nodes.
    """
    children = children_of(parents)[target]
    spouses = sorted({p for c in children for p in parents[c]} - {target})
    return {"parents": sorted(parents[target]), "children": children, "spouses": spouses}


def node_roles(parents, target):
    """The role of every node towards `target`: one of ROLES. A node can be several at once (a
    parent of the target that is also a parent of one of its children); the most direct wins:
    parent, then child, then spouse."""
    blanket = markov_blanket(parents, target)
    roles = ["outside"] * len(parents)
    for role, key in (("spouse", "spouses"), ("child", "children"), ("parent", "parents")):
        for node in blanket[key]:
            roles[node] = role
    roles[target] = "target"
    return roles


def node_info(parents):
    """The depth (longest path from a root), number of ancestors and of descendants of every node."""
    n = len(parents)
    depth, ancestors = [0] * n, [set() for _ in range(n)]
    for k, ps in enumerate(parents):
        depth[k] = 1 + max((depth[p] for p in ps), default=-1)
        ancestors[k] = set(ps).union(*(ancestors[p] for p in ps)) if ps else set()
    n_descendants = [sum(k in ancestors[j] for j in range(n)) for k in range(n)]
    return {"depth": depth, "n_ancestors": [len(a) for a in ancestors], "n_descendants": n_descendants}


def has_pure_spouse(parents, k):
    """Whether node `k` has a child and a spouse that is neither its parent nor its child, so
    that all four roles (parent, child, spouse, outside) exist around it."""
    blanket = markov_blanket(parents, k)
    pure = set(blanket["spouses"]) - set(blanket["parents"]) - set(blanket["children"])
    return bool(blanket["children"] and pure)
