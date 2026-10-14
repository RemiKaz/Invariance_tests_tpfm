"""The result files of run_exps.py, one json per (model, dataset, experiment).

`<results_dir>/runs/<model>/<dataset>/<exp>.json` holds every layer and metric computed for that
run, so a rerun only computes what is missing:

    {"model": ..., "dataset": ..., "exp": ...,
     "args": {the arguments that change the numbers, see KEY_ARGS},
     "test_accuracy": the clean test accuracy,
     "all_layers": the layers of `--layer all` once such a run is saved, else null,
     "layers": {layer: {metric: [one result per experiment of the group]}}}

A result is {"name", "title", "xlabel", "levels", "similarities"}, see run_experiment.

Only the standard library is used, so that the campaign scripts can ask what is left to run on
a login node, without the environment:

    python3 utils/results.py pending --models tabicl_v2 --datasets iris wine --exp all --metrics ccos nmi
    python3 utils/results.py accuracy --models tabicl_v2 --datasets iris wine --exp all
"""

import argparse
import json
import os
import re
import tempfile

# The arguments of run_exps.py that change the numbers: results computed with other values are
# not merged into the same file.
KEY_ARGS = ("model_path", "n_estimators", "feature_slot", "test_size", "n_levels", "seed", "split_seed")

# Named layers, in the order the data flows through the models that have them. Depths named
# by a fraction of the stack come first, then the layer indices, the stages and "last".
NAMED_ORDER = (
    "early",
    "middle",
    "tf_col_early",
    "tf_col_middle",
    "tf_col",
    "tf_col_1",
    "tf_row_1",
    "tf_col_2",
    "tf_row_2",
    "tf_row_early",
    "tf_row_middle",
    "tf_row",
    "tf_icl",
    "last",
)


def run_path(results_dir, model, dataset, exp):
    """The result file of one (model, dataset, experiment)."""
    return os.path.join(results_dir, "runs", model, dataset, f"{exp}.json")


def load(path):
    """The content of a result file, or None if there is none."""
    if not os.path.exists(path):
        return None
    with open(path) as file:
        return json.load(file)


def save(payload, path):
    """Write a result file atomically: a run killed while writing leaves the old file intact."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    with os.fdopen(fd, "w") as file:
        json.dump(payload, file, indent=1)
    os.replace(tmp, path)
    print(f"Saved results to {path}")


def missing(payload, layers, metrics):
    """layer -> the metrics of `metrics` without a result, for the layers missing any."""
    done = (payload or {}).get("layers", {})
    todo = {}
    for layer in layers:
        absent = [metric for metric in metrics if metric not in done.get(layer, {})]
        if absent:
            todo[layer] = absent
    return todo


def is_done(payload, metrics):
    """Whether a `--layer all` run is saved, with every metric of `metrics` for every layer."""
    return bool(payload and payload.get("all_layers")) and not missing(payload, payload["all_layers"], metrics)


def iter_runs(results_dir):
    """(model, dataset, exp, path) of every result file, in a stable order."""
    runs_dir = os.path.join(results_dir, "runs")
    if not os.path.isdir(runs_dir):
        return
    for model in sorted(os.listdir(runs_dir)):
        model_dir = os.path.join(runs_dir, model)
        if not os.path.isdir(model_dir):
            continue
        for dataset in sorted(os.listdir(model_dir)):
            dataset_dir = os.path.join(model_dir, dataset)
            for file_name in sorted(os.listdir(dataset_dir)) if os.path.isdir(dataset_dir) else ():
                if file_name.endswith(".json"):
                    yield model, dataset, file_name[: -len(".json")], os.path.join(dataset_dir, file_name)


def find_combinations(results_dir, exp=None, metric=None):
    """Every (model, dataset, exp, metric) with at least one result on disk, sorted.

    Args:
        results_dir (str): The directory of the run_exps.py results.
        exp (str, optional): Keep only this experiment (or group). Defaults to None, every one.
        metric (str, optional): Keep only this metric. Defaults to None, every one found.

    Returns:
        list: The (model, dataset, exp, metric) tuples.
    """
    combos = set()
    for model, dataset, run_exp, path in iter_runs(results_dir):
        if exp is not None and run_exp != exp:
            continue
        for metrics in load(path)["layers"].values():
            combos.update((model, dataset, run_exp, m) for m in metrics if metric is None or m == metric)
    return sorted(combos)


def layers_of(payload, metric):
    """layer -> the list of experiment results of one metric, for the layers that have it."""
    return {layer: metrics[metric] for layer, metrics in payload["layers"].items() if metric in metrics}


def layer_key(layer):
    """Sort key putting the layers in the order of the network."""
    if layer.lstrip("-").isdigit():
        return (2, int(layer), "")
    block = re.fullmatch(r"(col|row|icl)([12]?)_(\d+)", layer)
    if block:
        # the blocks of a stage come just before the stage's own output (tf_col_1 ...)
        stage = f"tf_{block.group(1)}" + (f"_{block.group(2)}" if block.group(2) else "")
        return (3 if stage == "tf_icl" else 1, NAMED_ORDER.index(stage) - 0.5, int(block.group(3)))
    if layer in NAMED_ORDER:
        # early / middle name a depth, then come the stages before the ICL blocks (TabPFN v3's
        # tf_col and tf_row), the indices, and the end of the model (tf_icl, last)
        rank = NAMED_ORDER.index(layer)
        group = 0 if rank < 2 else 3 if layer in ("tf_icl", "last") else 1
        return (group, rank, 0)
    return (4, 0, layer)  # a dotted submodule name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["pending", "accuracy"])
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--exp", default="all")
    parser.add_argument("--metrics", nargs="+", default=["cosine"], help="pending: the metrics a run needs")
    parser.add_argument("--results_dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
    args = parser.parse_args()

    for model in args.models:
        for dataset in args.datasets:
            payload = load(run_path(args.results_dir, model, dataset, args.exp))
            if args.command == "pending":
                if not is_done(payload, args.metrics):
                    print(model, dataset)  # one "model dataset" per line, read by scripts/config.sh
            else:
                accuracy = (payload or {}).get("test_accuracy")
                print(f"{model:12s} {dataset:42s} " + (f"{accuracy:.4f}" if accuracy is not None else "n/a (not run yet)"))


if __name__ == "__main__":
    main()
