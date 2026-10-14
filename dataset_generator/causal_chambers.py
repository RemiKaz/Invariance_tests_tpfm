"""Tables of real measurements with a known causal graph, from the Causal Chambers.

The Causal Chambers (Gamella, Peters & Bühlmann, "Causal chambers as a real-world physical
testbed for AI methodology", Nature Machine Intelligence 2025, https://causalchamber.org) are
two physical devices whose actuators and sensors are wired to a computer, with a ground-truth
causal graph established from first principles and validated by experiment. They are used
here to test, on real data, what the synthetic DAGs suggested: whether a model's embedding
reacts to the columns of the target's Markov blanket and not to the others.

Only the light tunnel in its standard configuration is used, and its dataset of independent
draws, `lt_interventions_standard_v1`. In its `uniform_reference` experiment (10 000 rows) the
light source (red, green, blue), the two polarisers (pol_1, pol_2) and the six LEDs next to
the sensors (l_11 ... l_32) are drawn independently and uniformly, and the sensor parameters
are fixed. The varying variables are those 11 actuators and 9 sensors (current, angle_1,
angle_2, ir_1 ... vis_3), whose parents are actuators: the graph has two layers. The wind
tunnel datasets are left out: they are time series of designed sequences (steps, impulses,
walks), whose dynamics (fan inertia) are not described by a graph between simultaneous
measurements.

A table is built for a target variable: `n_rows` rows of the experiment (disjoint subsets
across the tables of one experiment, `n_tables` of them), the target cut into `n_classes`
equal-frequency classes, the other varying variables as features, columns shuffled. The
variables kept constant by the experiment are dropped, which changes no conditional
independence between the others. Two graphs are stored (see
`datasets.causal_chamber_dataset.CausalChamberDataset`):

- `parents`: the published ground truth.
- `effective_parents`: its edges whose effect is measurable in the experiment, the partial R^2
  of the parent in a flexible regression of the child on its parents being at least
  `min_strength` (computed on every row of the experiment). In `uniform_reference` the LEDs
  l_ij have a published edge into the sensors of their sensor board but no visible effect
  (partial R^2 ~ 1e-4), so the Markov blanket of a variable through them is not identifiable
  from the data.

The default targets are the varying variables that have a non-empty effective Markov blanket
and at least two variables outside it (every one but the LEDs).

The dataset is downloaded once (3.9 MB, MD5-checked) into `<data_dir>/raw/`, which needs
internet: run the generation on a login node. Only numpy and scikit-learn are needed (no
`causalchamber` or pandas package): the ground-truth edges below are those of the
`causalchamber` package (causalchamber/ground_truth/main.py, MIT licence, Causal Chamber GmbH).

    python -m dataset_generator.causal_chambers                    # every default target, 3 tables each
    python -m dataset_generator.causal_chambers --targets current pol_1 --n_tables 5
    python -m dataset_generator.causal_chambers --list_experiments
"""

import argparse
import csv
import hashlib
import io
import json
import os
import urllib.request
import zipfile
from datetime import datetime, timezone

import numpy as np

from dataset_generator.scm_graph import markov_blanket, node_roles
from datasets.causal_chamber_dataset import causal_chamber_data_dir, causal_chamber_paths

GENERATOR_VERSION = 1
CITATION = (
    "Gamella, J. L., Peters, J., & Bühlmann, P. (2025). Causal chambers as a real-world physical "
    "testbed for AI methodology. Nature Machine Intelligence."
)

# --------------------------------------------------------------------------------------------
# Ground truth of the light tunnel, standard configuration (causalchamber package, MIT licence)

VARIABLES_LT_STANDARD = [
    "red", "green", "blue", "osr_c", "v_c", "current", "pol_1", "pol_2", "osr_angle_1",
    "osr_angle_2", "v_angle_1", "v_angle_2", "angle_1", "angle_2", "ir_1", "vis_1", "ir_2",
    "vis_2", "ir_3", "vis_3", "l_11", "l_12", "l_21", "l_22", "l_31", "l_32", "diode_ir_1",
    "diode_vis_1", "diode_ir_2", "diode_vis_2", "diode_ir_3", "diode_vis_3", "t_ir_1",
    "t_vis_1", "t_ir_2", "t_vis_2", "t_ir_3", "t_vis_3",
]

_RGB = ("red", "green", "blue")
EDGES_LT_STANDARD = (
    [(c, s) for s in ("ir_1", "ir_2", "ir_3", "vis_1", "vis_2", "vis_3", "current") for c in _RGB]
    + [("pol_1", "ir_3"), ("pol_2", "ir_3"), ("pol_1", "vis_3"), ("pol_2", "vis_3")]
    + [("pol_1", "angle_1"), ("v_angle_1", "angle_1"), ("osr_angle_1", "angle_1")]
    + [("pol_2", "angle_2"), ("v_angle_2", "angle_2"), ("osr_angle_2", "angle_2")]
    + [("v_c", "current"), ("osr_c", "current")]
    + [
        edge
        for board in ("1", "2", "3")
        for edge in (
            (f"l_{board}1", f"ir_{board}"), (f"l_{board}2", f"ir_{board}"),
            (f"l_{board}1", f"vis_{board}"), (f"l_{board}2", f"vis_{board}"),
            (f"t_ir_{board}", f"ir_{board}"), (f"diode_ir_{board}", f"ir_{board}"),
            (f"t_vis_{board}", f"vis_{board}"), (f"diode_vis_{board}", f"vis_{board}"),
        )
    ]
)

# name -> where to download it and which graph it follows
CHAMBER_DATASETS = {
    "lt_interventions_standard_v1": {
        "url": "https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables/lt_interventions_standard_v1.zip",
        "md5": "476664d024f88e8b7640998bb5e9ee33",
        "chamber": "lt",
        "configuration": "standard",
        "variables": VARIABLES_LT_STANDARD,
        "edges": EDGES_LT_STANDARD,
    },
}


# --------------------------------------------------------------------------------------------
# Download and reading


def download(dataset, data_dir=None):
    """The path of the dataset's zip, downloaded and checked against its MD5 if not there yet.

    Args:
        dataset (str): A key of CHAMBER_DATASETS.
        data_dir (str, optional): The data folder, the zip goes to its raw/. Defaults to
            `causal_chamber_data_dir()`.

    Returns:
        str: The path of the zip.

    Raises:
        ValueError: If the dataset is not implemented, or the download is corrupted.
    """
    if dataset not in CHAMBER_DATASETS:
        raise ValueError(f"Causal Chamber dataset {dataset} not implemented ! Available: {list(CHAMBER_DATASETS)}")
    info = CHAMBER_DATASETS[dataset]
    raw_dir = os.path.join(data_dir or causal_chamber_data_dir(), "raw")
    path = os.path.join(raw_dir, f"{dataset}.zip")
    if not os.path.exists(path):
        os.makedirs(raw_dir, exist_ok=True)
        print(f"Downloading {info['url']} (needs internet: a login node)")
        tmp = path + ".part"
        urllib.request.urlretrieve(info["url"], tmp)
        os.replace(tmp, path)
    with open(path, "rb") as file:
        md5 = hashlib.md5(file.read()).hexdigest()
    if md5 != info["md5"]:
        raise ValueError(f"{path} has MD5 {md5}, expected {info['md5']}: delete it and download again")
    return path


def list_experiments(dataset, data_dir=None):
    """The experiments of a dataset (one csv each in its zip)."""
    with zipfile.ZipFile(download(dataset, data_dir)) as archive:
        return sorted(os.path.basename(n)[: -len(".csv")] for n in archive.namelist() if n.endswith(".csv"))


def read_experiment(dataset, experiment, data_dir=None):
    """The graph variables of one experiment, as floats.

    Args:
        dataset (str): A key of CHAMBER_DATASETS.
        experiment (str): The experiment, e.g. uniform_reference.
        data_dir (str, optional): The data folder. Defaults to `causal_chamber_data_dir()`.

    Returns:
        np.ndarray: The values, of shape (n_rows, n_variables), columns in the order of the
            dataset's `variables` (the bookkeeping columns, timestamp, flag, ..., are dropped).
    """
    variables = CHAMBER_DATASETS[dataset]["variables"]
    with zipfile.ZipFile(download(dataset, data_dir)) as archive:
        member = f"{dataset}/{experiment}.csv"
        if member not in archive.namelist():
            raise ValueError(f"No experiment {experiment} in {dataset} ! Available: {list_experiments(dataset, data_dir)}")
        rows = list(csv.reader(io.StringIO(archive.read(member).decode())))
    header = rows[0]
    columns = [header.index(v) for v in variables]
    return np.array([[float(row[c]) for c in columns] for row in rows[1:]])


# --------------------------------------------------------------------------------------------
# The graph of the varying variables, and how strong its edges are in the data


def topological_order(variables, edges):
    """The variables sorted so that every parent comes before its children (the order of the
    nodes of a stored table), ties broken by the published order."""
    parents = {v: {p for p, c in edges if c == v} for v in variables}
    order, placed = [], set()
    while len(order) < len(variables):
        ready = [v for v in variables if v not in placed and parents[v] <= placed]
        if not ready:
            raise ValueError("the ground-truth graph has a cycle")
        order.extend(ready)
        placed.update(ready)
    return order


def edge_strengths(values, parents):
    """The partial R^2 of every edge: how much the R^2 of a flexible regression of the child on
    its parents drops when that parent is left out.

    The regression has a cubic spline of each parent and every pairwise product of parents, so
    that it captures the non-linear effects of the chamber (Malus's law makes the light at the
    third sensor board depend on cos^2 of the difference of the two polariser angles).

    Args:
        values (np.ndarray): The nodes, of shape (n_rows, n_nodes).
        parents (list): The parents of every node.

    Returns:
        list: One {"parent", "child", "partial_r2"} per edge (node indices).
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import SplineTransformer

    def r2(child, cols):
        if not cols:
            return 0.0
        Z = (values[:, cols] - values[:, cols].mean(0)) / values[:, cols].std(0)
        design = [SplineTransformer(n_knots=5, degree=3).fit_transform(Z[:, [i]]) for i in range(len(cols))]
        design += [Z[:, [i]] * Z[:, [j]] for i in range(len(cols)) for j in range(i + 1, len(cols))]
        X = np.hstack(design)
        return LinearRegression().fit(X, values[:, child]).score(X, values[:, child])

    out = []
    for child, ps in enumerate(parents):
        if ps:
            full = r2(child, ps)
            out += [{"parent": p, "child": child, "partial_r2": full - r2(child, [q for q in ps if q != p])} for p in ps]
    return out


def experiment_graph(dataset, experiment, min_strength=1e-3, data_dir=None):
    """The varying variables of an experiment in topological order, their values, the published
    graph between them and its measurable part.

    Args:
        dataset (str): A key of CHAMBER_DATASETS.
        experiment (str): The experiment.
        min_strength (float, optional): The partial R^2 an edge needs to be kept in the
            effective graph. Defaults to 1e-3.
        data_dir (str, optional): The data folder. Defaults to `causal_chamber_data_dir()`.

    Returns:
        dict: "variables" (names, node order), "values" (n_rows, n_nodes), "parents",
            "effective_parents", "edge_strength" and "constant" (the variables dropped).
    """
    info = CHAMBER_DATASETS[dataset]
    raw = read_experiment(dataset, experiment, data_dir)
    varying = [v for v, column in zip(info["variables"], raw.T) if column.std() > 0]
    variables = topological_order(varying, [e for e in info["edges"] if e[0] in varying and e[1] in varying])
    values = raw[:, [info["variables"].index(v) for v in variables]]
    parents = [sorted(variables.index(p) for p, c in info["edges"] if c == v and p in variables) for v in variables]
    # the regressions take a few seconds and every --names_only call (each stage of
    # scripts/pipeline.sh) needs the effective graph, so they are cached next to the zip
    cache = os.path.join(data_dir or causal_chamber_data_dir(), "raw", f"{dataset}__{experiment}__edges.json")
    if os.path.exists(cache):
        with open(cache) as file:
            cached = json.load(file)
        strength = cached["edge_strength"] if cached["variables"] == variables else None
    else:
        strength = None
    if strength is None:
        strength = edge_strengths(values, parents)
        with open(cache, "w") as file:
            json.dump({"variables": variables, "edge_strength": strength}, file, indent=1)
    effective = [[] for _ in variables]
    for edge in strength:
        if edge["partial_r2"] >= min_strength:
            effective[edge["child"]].append(edge["parent"])
    return {
        "variables": variables,
        "values": values,
        "parents": parents,
        "effective_parents": [sorted(p) for p in effective],
        "edge_strength": strength,
        "constant": [v for v in info["variables"] if v not in varying],
    }


def default_targets(graph, min_outside=2):
    """The variables whose effective Markov blanket is not empty, with at least `min_outside`
    variables outside it in both graphs."""
    targets = []
    for t, name in enumerate(graph["variables"]):
        ok = True
        for key in ("parents", "effective_parents"):
            roles = node_roles(graph[key], t)
            ok &= roles.count("outside") >= min_outside
        ok &= any(r != "outside" for i, r in enumerate(node_roles(graph["effective_parents"], t)) if i != t)
        if ok:
            targets.append(name)
    return targets


# --------------------------------------------------------------------------------------------
# Tables


def table_name(chamber, experiment, target, index):
    """cc_lt_ref_current_00 for the first table of target `current` in `uniform_reference`."""
    short = "ref" if experiment == "uniform_reference" else experiment.removeprefix("uniform_")
    return f"cc_{chamber}_{short}_{target}_{index:02d}"


def build_table(graph, target, rows, n_classes, rng, permute_columns=True):
    """One table: the rows `rows` of the experiment, the target cut into classes.

    Returns:
        tuple: X, y, the nodes (target included) and the metadata that depends on the table.
    """
    t = graph["variables"].index(target)
    nodes = graph["values"][rows]
    thresholds = np.quantile(nodes[:, t], np.arange(1, n_classes) / n_classes)
    y = np.searchsorted(thresholds, nodes[:, t]).astype(np.int64)
    if len(np.unique(y)) < n_classes:
        raise ValueError(f"{target} has too few distinct values for {n_classes} classes")
    feature_nodes = [k for k in range(len(graph["variables"])) if k != t]
    if permute_columns:
        feature_nodes = [feature_nodes[i] for i in rng.permutation(len(feature_nodes))]
    blankets = {}
    for key, truth in (("parents", "graph"), ("effective_parents", "effective")):
        blanket = markov_blanket(graph[key], t)
        blankets[truth] = {role: [graph["variables"][k] for k in nodes_] for role, nodes_ in blanket.items()}
    metadata = {
        "target_node": t,
        "target_variable": target,
        "feature_nodes": feature_nodes,
        "target_thresholds": thresholds.tolist(),
        "class_counts": np.bincount(y, minlength=n_classes).tolist(),
        "markov_blanket": blankets,
        "node_roles": {"graph": node_roles(graph["parents"], t), "effective": node_roles(graph["effective_parents"], t)},
    }
    return nodes[:, feature_nodes], y, nodes, metadata


def store(name, X, y, nodes, metadata, data_dir=None):
    """Write `<name>.npz` (X, y, nodes) and `<name>.json` to the data folder."""
    table_path, metadata_path = causal_chamber_paths(name, data_dir)
    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    np.savez(table_path, X=X, y=y, nodes=nodes)
    with open(metadata_path, "w") as file:
        json.dump({"name": name, **metadata}, file, indent=2)
    print(f"Stored {name}: X {X.shape}, classes {metadata['class_counts']}, "
          f"effective blanket {sum(len(v) for v in metadata['markov_blanket']['effective'].values())} nodes")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="lt_interventions_standard_v1", choices=list(CHAMBER_DATASETS))
    parser.add_argument("--experiment", default="uniform_reference", help="see --list_experiments")
    parser.add_argument("--targets", nargs="+", default=["auto"],
                        help="target variables, 'auto' for every one with a measurable Markov blanket")
    parser.add_argument("--n_tables", type=int, default=3, help="tables per target, on disjoint rows")
    parser.add_argument("--n_rows", type=int, default=1000, help="rows of a table (the context is 67%% of them)")
    parser.add_argument("--n_classes", type=int, default=3)
    parser.add_argument("--min_strength", type=float, default=1e-3,
                        help="partial R^2 an edge needs to be in the effective graph")
    parser.add_argument("--no_permute_columns", action="store_true", help="keep the feature columns in node order")
    parser.add_argument("--seed", type=int, default=0, help="seed of the row subsets and column orders")
    parser.add_argument("--data_dir", help="default: $CAUSAL_CHAMBER_DATA or the lustre folder")
    parser.add_argument("--skip_existing", action="store_true", help="leave a table that is already stored alone")
    parser.add_argument("--names_only", action="store_true", help="only print the table names, nothing is stored")
    parser.add_argument("--list_experiments", action="store_true", help="print the experiments of --dataset")
    args = parser.parse_args()

    if args.list_experiments:
        print("\n".join(list_experiments(args.dataset, args.data_dir)))
        return
    info = CHAMBER_DATASETS[args.dataset]
    graph = experiment_graph(args.dataset, args.experiment, args.min_strength, args.data_dir)
    targets = default_targets(graph) if args.targets == ["auto"] else args.targets
    unknown = [t for t in targets if t not in graph["variables"]]
    if unknown:
        raise SystemExit(f"Targets {unknown} do not vary in {args.experiment} ! Available: {graph['variables']}")
    n_rows_total = len(graph["values"])
    if args.n_tables * args.n_rows > n_rows_total:
        raise SystemExit(f"{args.experiment} has {n_rows_total} rows, not enough for {args.n_tables} disjoint tables of {args.n_rows}")
    if not args.names_only:
        print(f"{args.experiment}: {n_rows_total} rows, {len(graph['variables'])} varying variables, "
              f"constant (dropped): {graph['constant']}")
        weak = [e for e in graph["edge_strength"] if e["partial_r2"] < args.min_strength]
        print(f"Published edges without a measurable effect (partial R^2 < {args.min_strength}): "
              + ", ".join(f"{graph['variables'][e['parent']]}->{graph['variables'][e['child']]}" for e in weak))

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_rows_total)  # table k of every target takes the same rows
    for index in range(args.n_tables):
        rows = np.sort(order[index * args.n_rows : (index + 1) * args.n_rows])
        for target in targets:
            name = table_name(info["chamber"], args.experiment, target, index)
            if args.names_only:
                print(name)
                continue
            if args.skip_existing and os.path.exists(causal_chamber_paths(name, args.data_dir)[0]):
                print(f"Skipping {name}, already stored")
                continue
            table_rng = np.random.default_rng([args.seed, index, graph["variables"].index(target)])
            X, y, nodes, metadata = build_table(graph, target, rows, args.n_classes, table_rng,
                                                permute_columns=not args.no_permute_columns)
            metadata = {
                "generator": "causal_chambers",
                "generator_version": GENERATOR_VERSION,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "structure": "causal_chamber",
                "source": {"dataset": args.dataset, "experiment": args.experiment, "url": info["url"],
                           "md5": info["md5"], "chamber": info["chamber"],
                           "configuration": info["configuration"], "citation": CITATION},
                "config": {"n_rows": args.n_rows, "n_classes": args.n_classes, "seed": args.seed,
                           "table_index": index, "permute_columns": not args.no_permute_columns,
                           "min_strength": args.min_strength},
                "n_samples": len(y),
                "n_nodes": len(graph["variables"]),
                "variables": graph["variables"],
                "parents": graph["parents"],
                "effective_parents": graph["effective_parents"],
                "edge_strength": graph["edge_strength"],
                "constant_variables": graph["constant"],
                "rows": rows.tolist(),
                **metadata,
                "table_arrays": {"X": "features, columns are feature_nodes",
                                 "y": "integer labels of the target node",
                                 "nodes": "every varying variable, columns in node order (see variables)"},
            }
            store(name, X, y, nodes, metadata, args.data_dir)


if __name__ == "__main__":
    main()
