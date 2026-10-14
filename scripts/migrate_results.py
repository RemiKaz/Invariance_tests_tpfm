"""One-off: convert the results of the old layout to the new one, then this file can go.

Old: results/runs/<model>_<dataset>_<exp>_layer-<layer>[_metric-<metric>]/<same>.json, one file
per (layer, metric). New: results/runs/<model>/<dataset>/<exp>.json, see utils/results.py.

    python3 scripts/migrate_results.py --dry_run   # report only
    python3 scripts/migrate_results.py             # write the new files, the old ones are kept

The metrics no longer computed (mi, cka) are left out. A (model, dataset, exp) whose files were
computed with different arguments (KEY_ARGS) is reported and not converted. A run is marked
complete (all_layers) when every layer the campaign of the time read for its model is there.
Once the figures look right, the old folders can be removed: they are the ones of
`ls -d results/runs/*_layer-*`.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from results import KEY_ARGS, run_path, save  # noqa: E402 - stdlib only, no environment needed

KEEP = ("cosine", "ccos", "nmi", "dcka")
# The defaults of the old run_exps.py, for the files written before an argument existed.
OLD_DEFAULTS = {"model_path": None, "n_estimators": 1, "feature_slot": "target", "test_size": 0.33,
                "n_levels": 20, "seed": 0, "split_seed": 42}
# layers_of of the old scripts/config.sh
_ICL24 = [str(i) for i in range(24)]
OLD_LAYERS = {
    "tabicl_v1": ["col_0", "col_1", "tf_col", "row_0", "row_1", "tf_row"] + [f"icl_{i}" for i in range(12)] + ["tf_icl"],
    "tabfm_v1": ["col1_0", "col1_1", "tf_col_1", "row1_0", "row1_1", "row1_2", "tf_row_1",
                 "col2_0", "col2_1", "tf_col_2", "row2_0", "row2_1", "row2_2", "tf_row_2"] + _ICL24 + ["tf_icl"],
    "tabpfn_v3": ["col_0", "col_1", "tf_col", "row_0", "row_1", "tf_row"] + _ICL24 + ["tf_icl", "last"],
    "tabpfn_v2": [str(i) for i in range(12)] + ["last"],
    "limix_v2": _ICL24 + ["last"],
}
OLD_LAYERS["tabicl_v2"] = OLD_LAYERS["tabicl_v1"]
OLD_LAYERS["mitra_v1"] = OLD_LAYERS["mitra_v2"] = OLD_LAYERS["tabpfn_v2"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="replace new-layout files that already exist")
    args = parser.parse_args()

    runs_dir = os.path.join(args.results_dir, "runs")
    groups = defaultdict(list)  # (model, dataset, exp) -> [(args, payload)]
    n_files = n_dropped = 0
    for folder in sorted(os.listdir(runs_dir)):
        path = os.path.join(runs_dir, folder, f"{folder}.json")
        if "_layer-" not in folder or not os.path.exists(path):
            continue
        with open(path) as file:
            payload = json.load(file)
        run_args = payload["args"]
        n_files += 1
        if run_args.get("metric", "cosine") not in KEEP:
            n_dropped += 1
            continue
        groups[(run_args["model"], run_args["dataset"], run_args["exp"])].append((run_args, payload))
    print(f"{n_files} old result files, {n_dropped} of a metric no longer computed left out, {len(groups)} runs")

    n_written = 0
    for (model, dataset, exp), files in sorted(groups.items()):
        key_args = [
            {name: (None if run_args.get(name) == "default" else run_args.get(name, OLD_DEFAULTS[name])) for name in KEY_ARGS}
            for run_args, _ in files
        ]
        if any(k != key_args[0] for k in key_args):
            variants = sorted({json.dumps(k, sort_keys=True) for k in key_args})
            print(f"SKIPPED {model} {dataset} {exp}: files computed with different arguments: {variants}")
            continue
        layers = defaultdict(dict)
        accuracy = None
        for run_args, payload in files:
            results = [{k: v for k, v in r.items() if k != "metric"} for r in payload["results"]]
            layers[str(run_args["layer"])][run_args.get("metric", "cosine")] = results
            accuracy = payload.get("test_accuracy", accuracy)
        expected = OLD_LAYERS.get(model)
        new = {
            "model": model, "dataset": dataset, "exp": exp, "args": key_args[0], "test_accuracy": accuracy,
            "all_layers": expected if expected and all(layer in layers for layer in expected) else None,
            "layers": dict(layers),
        }
        path = run_path(args.results_dir, model, dataset, exp)
        if os.path.exists(path) and not args.overwrite:
            print(f"EXISTS {path}, left alone (--overwrite replaces it)")
            continue
        if args.dry_run:
            print(f"would write {path}: {len(layers)} layers, complete={new['all_layers'] is not None}")
        else:
            save(new, path)
        n_written += 1
    print(f"{n_written} runs {'to write' if args.dry_run else 'written'}")


if __name__ == "__main__":
    main()
