"""Fuse the results of a model on a dataset into a single figure, one curve per layer.

Every run of `run_exps.py` writes one json file per (model, dataset, experiment), holding every
layer and metric (see utils/results.py). This script draws, for one metric, a panel per
experiment with one curve per layer, so the layers can be compared on the same axes.

Examples:
    python fuse_plot.py --model tabicl_v2 --dataset breast_cancer --exp all --metric ccos
    python fuse_plot.py --model mitra_v2 --dataset breast_cancer --layers early middle last
    python fuse_plot.py --all                    # every (model, dataset, exp, metric) found
    python fuse_plot.py --all --exp all           # only the "all" experiment group

The figure is written to `<results_dir>/_fused/<model>_<dataset>_<exp>_<metric>_layers.png`
(`_roles.png` with --by_role), and with --csv a csv of the plotted numbers next to it. A figure
that is already newer than its result file, the table it takes the roles from, this script and
utils/plots.py is not drawn again; --force draws it anyway.
"""

import argparse
import csv
import json
import os

from dataset_generator.scm_graph import graph_of, node_roles
from datasets.causal_chamber_dataset import causal_chamber_paths
from datasets.custom_scm_dataset import custom_scm_paths
from utils.metrics import metric_label
from utils.plots import MODULAR_MODELS, plot_layers, plot_roles
from utils.results import find_combinations, layer_key, layers_of, load, run_path

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(HERE, "results")
CODE_FILES = [os.path.join(HERE, "fuse_plot.py"), os.path.join(HERE, "utils", "plots.py")]  # a restyled figure is stale


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", default=None, help="Model of the runs, e.g. tabicl_v2")
    parser.add_argument("--dataset", default=None, help="Dataset of the runs, e.g. breast_cancer")
    parser.add_argument(
        "--exp",
        default=None,
        help='Experiment (or group) of the runs. Default: "all" for a single plot, every '
        "experiment found for --all.",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help='Metric of the runs (see utils/metrics.py). Default: "cosine" for a single plot, every '
        "metric found for --all. Metrics are never mixed into the same figure, their scales differ.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fuse every (model, dataset, exp, metric) combination found in --results_dir instead "
        "of one plot for --model/--dataset. Incompatible with --layers and --out.",
    )
    parser.add_argument(
        "--results_dir", default=DEFAULT_RESULTS_DIR, help="Directory holding runs/ (the run_exps.py results); the figures go to its _fused/"
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=None,
        help="Layers to draw, in this order. Default: every layer found, by depth. Set it when\n"
        "mixing depth names (early, middle) with layer indices, whose depths cannot be compared.",
    )
    parser.add_argument(
        "--full_scale",
        action="store_true",
        help="Draw the y axis from 0 to 1 instead of zooming on the range of the curves",
    )
    parser.add_argument(
        "--csv", action="store_true", help="Also write the plotted numbers, as a csv next to each figure"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Draw the figures again even when they are newer than their result files",
    )
    parser.add_argument(
        "--by_role",
        action="store_true",
        help="Instead of one curve per layer, draw one curve per role of the perturbed node "
        "towards the target (target, parent, child, spouse, outside its Markov blanket) with "
        "the layers on the x axis; needs a table stored by dataset_generator",
    )
    parser.add_argument("--out", default=None, help="Path of the figure (png), single-plot mode only")
    args = parser.parse_args()

    if args.all:
        if args.layers is not None or args.out is not None:
            parser.error("--all cannot be combined with --layers or --out")
    elif args.model is None or args.dataset is None:
        parser.error("--model and --dataset are required (or pass --all)")
    return args


def figure_path(results_dir, model, dataset, exp, metric, by_role=False):
    """Where the fused figure of a combination is written."""
    kind = "roles" if by_role else "layers"
    return os.path.join(results_dir, "_fused", f"{model}_{dataset}_{exp}_{metric}_{kind}.png")


def up_to_date(outputs, inputs):
    """Whether every output exists and is newer than every input that exists."""
    if not all(os.path.exists(path) for path in outputs):
        return False
    oldest = min(os.path.getmtime(path) for path in outputs)
    return all(os.path.getmtime(path) <= oldest for path in inputs if os.path.exists(path))


def table_metadata_path(dataset):
    """The json of a table stored by dataset_generator, else of a Causal Chamber table (which may
    not exist either)."""
    path = custom_scm_paths(dataset)[1]
    return path if os.path.exists(path) else causal_chamber_paths(dataset)[1]


def table_roles(dataset):
    """The role of every node of a table stored by dataset_generator (or of a Causal Chamber
    table, in its published graph) towards its target, or None if there is no such table."""
    metadata_path = table_metadata_path(dataset)
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path) as file:
        metadata = json.load(file)
    return node_roles(graph_of(metadata)["parents"], metadata["target_node"])


def write_table(runs, layers, out_path):
    """The plotted numbers, one row per point, next to the figure."""
    with open(out_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["experiment", "layer", "level", "value"])
        for layer in layers:
            for result in runs[layer]:
                for level, similarity in zip(result["levels"], result["similarities"]):
                    writer.writerow([result["name"], layer, level, similarity])


def fuse_one(
    results_dir, model, dataset, exp, metric, full_scale, out_path=None, csv_too=False,
    by_role=False, layers=None, force=False,
):
    """Load, plot and save the fused figure of one combination, skipped-on-error.

    The figure is not drawn again when it is newer than the result file it comes from, the
    table it takes the roles from and the plotting code, unless `force`, or the options change
    what is drawn (a choice of `layers`, `full_scale`), in which case an older figure of the
    same name may be of something else.

    Returns:
        list or None: The layers drawn, or None if the combination had no usable result.
    """
    label = f"  {model} {dataset} {exp} {metric}:"
    out_path = out_path or figure_path(results_dir, model, dataset, exp, metric, by_role)
    path = run_path(results_dir, model, dataset, exp)
    payload = load(path)
    runs = layers_of(payload, metric) if payload else {}
    if not runs:
        print(f"{label} skipped, no result in {path}")
        return None
    found = sorted(runs, key=layer_key)
    layers = layers if layers is not None else found
    missing = [layer for layer in layers if layer not in runs]
    if missing:
        print(f"{label} skipped, no result for layers {missing}, found: {found}")
        return None

    if not force and layers == found and not full_scale:
        inputs = [path] + CODE_FILES + ([table_metadata_path(dataset)] if by_role else [])
        outputs = [out_path] + ([os.path.splitext(out_path)[0] + ".csv"] if csv_too and not by_role else [])
        if up_to_date(outputs, inputs):
            print(f"{label} up to date, not drawn again (--force draws it)")
            return layers

    if by_role:
        roles = table_roles(dataset)
        if roles is None:
            print(f"{label} skipped, no stored table of that name")
            return None
        title = f"{model} / {dataset} / {exp}: {metric_label(metric)} by role of the perturbed node"
        plot_roles(runs, layers, roles, title, out_path, metric, full_scale)
        return layers
    title = f"{model} / {dataset} / {exp}: {metric_label(metric)} by layer"
    plot_layers(runs, layers, title, out_path, metric, full_scale, modular=model in MODULAR_MODELS)
    if csv_too:
        write_table(runs, layers, os.path.splitext(out_path)[0] + ".csv")
    return layers


def main():
    args = parse_args()

    if args.all:
        combos = find_combinations(args.results_dir, args.exp, args.metric)
        if not combos:
            filters = [f" exp={args.exp}" if args.exp else "", f" metric={args.metric}" if args.metric else ""]
            raise SystemExit(f"No result found in {args.results_dir} for{''.join(filters) or ' any'} combination")
        print(f"{len(combos)} (model, dataset, exp, metric) combination(s) found")
        n_done = 0
        for model, dataset, exp, metric in combos:
            layers = fuse_one(
                args.results_dir, model, dataset, exp, metric, args.full_scale, csv_too=args.csv,
                by_role=args.by_role, force=args.force,
            )
            n_done += layers is not None
        print(f"{n_done}/{len(combos)} plots are in {os.path.join(args.results_dir, '_fused')}")
        return

    exp = args.exp or "all"
    metric = args.metric or "cosine"
    layers = fuse_one(
        args.results_dir, args.model, args.dataset, exp, metric, args.full_scale, out_path=args.out,
        csv_too=args.csv, by_role=args.by_role, layers=args.layers, force=args.force,
    )
    if layers is None:
        raise SystemExit(1)
    out_path = args.out or figure_path(args.results_dir, args.model, args.dataset, exp, metric, args.by_role)
    print(f"Layers {layers}\nFigure: {out_path}")


if __name__ == "__main__":
    main()
