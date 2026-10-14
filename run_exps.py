"""Run invariance experiments on a tabular foundation model.

Every layer and metric of a (model, dataset, experiment) is saved to one file,
`<results_dir>/runs/<model>/<dataset>/<exp>.json` (see utils/results.py), and a rerun only
computes the (layer, metric) pairs it does not hold yet; --force recomputes everything.

Example:
    python run_exps.py --model tabpfn_v3 --dataset breast_cancer --exp perturbations --layer all
"""

import argparse

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from datasets import dataset_importer
from experiments import EXPERIMENT_GROUPS, EXPERIMENTS, experiment_importer, run_experiment
from models import MODEL_NAMES, model_importer
from utils import METRICS
from utils.results import KEY_ARGS, load, missing, run_path, save


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model", type=str, default="tabpfn_v3", choices=list(MODEL_NAMES), help="Model to probe"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="breast_cancer",
        help="Dataset to run the experiments on: breast_cancer, iris, wine, digits, phoneme, "
        "spambase, or the name of a table stored by dataset_generator (--exp chain needs one)",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="perturbations",
        choices=list(EXPERIMENTS) + list(EXPERIMENT_GROUPS),
        help="Experiment, or group of experiments, to run",
    )
    parser.add_argument(
        "--layer",
        type=str,
        nargs="+",
        default=["all"],
        help="Layer positions the embeddings are read at: 'all' for every block and stage of the "
        "model (see --list_layers), 'last' for the end of the model, an index in the layer "
        "stack, a stage or block name, or a dotted submodule name. They are all read from the "
        "same forward passes.",
    )
    parser.add_argument(
        "--list_layers",
        action="store_true",
        help="Print the layers the embeddings can be read at, then exit",
    )
    parser.add_argument(
        "--metric",
        type=str,
        nargs="+",
        default=["cosine"],
        choices=list(METRICS),
        help="How the perturbed embeddings are compared to the baseline, see utils/metrics.py. "
        "Several can be given, they are all computed from the same embeddings (one sweep).",
    )

    parser.add_argument("--model_path", type=str, default=None, help="Checkpoint path, default: the model's own")
    parser.add_argument("--n_estimators", type=int, default=1, help="Size of the model ensemble")
    parser.add_argument(
        "--feature_slot",
        type=str,
        default="target",
        choices=["target", "mean", "flatten"],
        help="How per-cell activations are reduced to one vector per row",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")

    parser.add_argument("--test_size", type=float, default=0.33, help="Test set proportion")
    parser.add_argument("--n_levels", type=int, default=20, help="Number of intensity steps")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the perturbations")
    parser.add_argument("--split_seed", type=int, default=42, help="Seed of the train/test split")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory, runs go to its runs/")
    parser.add_argument("--force", action="store_true", help="Recompute the results already saved")
    parser.add_argument(
        "--eval", action="store_true", help="Also report the ROC AUC on the clean data"
    )
    return parser.parse_args()


def print_layers(model):
    layers = model.list_layers()
    print("\nDefault layers (--layer all):\n  " + " ".join(layers["default"]))
    print("\nLayer stack (addressable by index):")
    if not layers["layer_stack"]:
        print("  none found for this model, address a submodule by name")
    for index, name in layers["layer_stack"]:
        print(f"  {index:3d}  {name}")
    for depth, name in layers["depths"].items():
        print(f"  {depth:>6s}  = {name}")
    print("\nAll submodules (addressable by name):")
    for name in layers["all_modules"]:
        print(f"       {name}")


def main():
    args = parse_args()

    model = model_importer(
        args.model,
        device=args.device,
        model_path=args.model_path,
        n_estimators=args.n_estimators,
        feature_slot=args.feature_slot,
        seed=args.seed,
    )
    dataset = dataset_importer(args.dataset, test_size=args.test_size, seed=args.split_seed)
    model.fit(dataset.X_train, dataset.y_train)
    if args.list_layers:
        print_layers(model)
        return

    asked = list(dict.fromkeys(args.layer))  # drop duplicates, keep order
    if "all" in asked:
        index = asked.index("all")
        asked[index : index + 1] = model.default_layers()
    layers = list(dict.fromkeys(asked))
    metrics = list(dict.fromkeys(args.metric))

    path = run_path(args.results_dir, args.model, args.dataset, args.exp)
    key_args = {name: getattr(args, name) for name in KEY_ARGS}
    payload = None if args.force else load(path)
    if payload is not None and payload["args"] != key_args:
        raise SystemExit(
            f"{path} holds results computed with {payload['args']}, not {key_args}: --force "
            "replaces them, or pass another --results_dir"
        )
    payload = payload or {
        "model": args.model, "dataset": args.dataset, "exp": args.exp, "args": key_args,
        "all_layers": None, "layers": {},
    }
    if "all" in args.layer:
        payload["all_layers"] = model.default_layers()

    todo = missing(payload, layers, metrics)
    if not todo:
        print(f"Nothing to compute, every layer and metric asked is in {path} (--force recomputes them)")
        save(payload, path)  # all_layers may be new
        return
    todo_metrics = [metric for metric in metrics if any(metric in absent for absent in todo.values())]
    print(f"To compute: layers {list(todo)}, metrics {todo_metrics}")

    # the clean test accuracy, saved with the curves: it is what a perturbation is ultimately
    # expected to preserve, and puts the curves of different datasets in context
    probabilities = model.predict_proba(dataset.X_test)
    payload["test_accuracy"] = accuracy_score(dataset.y_test, model.predict(dataset.X_test))
    print(f"Test accuracy on the clean data: {payload['test_accuracy']:.4f}")
    if args.eval:
        if probabilities.shape[1] == 2:
            print("ROC AUC:", roc_auc_score(dataset.y_test, probabilities[:, 1]))
        else:
            print("ROC AUC:", roc_auc_score(dataset.y_test, probabilities, multi_class="ovr"))

    # The baseline is read once: every experiment compares its perturbed embeddings to the
    # ones of the same clean fit.
    baseline, failed = model.get_embeddings_of(dataset.X_test, list(todo))
    for layer, error in failed.items():
        print(f"FAILED layer {layer}: {type(error).__name__}: {error}")
    if not baseline:
        raise SystemExit("FAILED: no layer could be read")
    rng = np.random.default_rng(args.seed)
    by_experiment = [
        run_experiment(experiment, model, dataset, baseline, rng=rng, n_levels=args.n_levels, metrics=todo_metrics)
        for experiment in experiment_importer(args.exp)
    ]
    for layer in baseline:
        for metric in todo[layer]:
            payload["layers"].setdefault(layer, {})[metric] = [results[layer][metric] for results in by_experiment]
    save(payload, path)
    if failed:
        print(f"FAILED layers: {list(failed)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
