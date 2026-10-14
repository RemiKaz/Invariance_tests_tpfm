"""Markov blanket identification benchmark: tabular foundation models against classical methods.

One run evaluates every detector of detectors/ on every table matched by --tables (the Causal
Chamber tables of dataset_generator/causal_chambers.py, or the simulated DAGs), and writes a
leaderboard. Each detector says which feature columns are in the Markov blanket of the target:

- the models (`pfm`), through their embedding: how much perturbing one column alone moves the
  final-layer embedding of the test rows (--layers final), or their prediction: how much it moves
  the predicted class probabilities (--layers output, total variation distance). Each model is
  loaded once and run here on every table (one fit per column and perturbation, the scores cached
  in <results_dir>/blanket_scores/), so a GPU is needed; --source results reads the curves of
  `run_exps.py --exp blanket` instead.
- classical methods, fitted to the same context rows on the CPU: IAMB, Grow-Shrink, total
  conditioning, and the marginal association of each column with the target (not a blanket
  method, the reference to beat), each with a conditional independence test (fisher_z,
  logistic, spline).

Every detector is scored as a ranking (AUC of the blanket columns against the columns outside
it, precision of its top k when k is the true blanket size) and as a set (precision, recall, F1,
exact recovery, recall by role, share of outside columns selected), on every table, then
averaged over the tables. A model has no threshold of its own: its set is cut at the largest gap
of its scores (detectors/base_detector.py).

The truth is the published graph (`graph`) and, for the Causal Chamber tables, its edges with a
measurable effect in the data (`effective`, see datasets/causal_chamber_dataset.py), the one the
leaderboard is sorted by.

    python benchmark_blanket.py --tables 'cc_lt_ref_*' --models tabpfn_v3 --n_jobs 8   # in a GPU job
    python benchmark_blanket.py --tables 'cc_lt_ref_*' --models tabpfn_v3 --layers final output
    python benchmark_blanket.py --tables 'cc_lt_ref_*' --models tabpfn_v3 --probe   # + the embedding variants
    python benchmark_blanket.py --tables 'dagpool_*' --models limix_v2 --source results --truths graph

Writes to <results_dir>/benchmark/: blanket_<tag>_leaderboard.md, blanket_<tag>_summary.csv (mean
and s.e. of every metric), blanket_<tag>_by_target.csv and blanket_<tag>.csv (one row per table,
detector and truth).
"""

import argparse
import csv
import fnmatch
import json
import os
from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed

from dataset_generator.scm_graph import graph_of, node_roles
from datasets import dataset_importer
from datasets.causal_chamber_dataset import causal_chamber_data_dir
from datasets.custom_scm_dataset import custom_scm_data_dir
from detectors import detector_importer
from detectors.pfm_perturbation import release_models
from detectors.pfm_probe import probe_grid

HERE = os.path.dirname(os.path.abspath(__file__))
BLANKET = ("parent", "child", "spouse")
CLASSICAL = ["marginal:logistic", "total_conditioning:logistic", "total_conditioning:spline", "iamb:fisher_z",
             "iamb:logistic", "iamb:spline", "grow_shrink:logistic", "grow_shrink:spline"]
METRICS = ("auc", "topk_precision", "topk_exact", "precision", "recall", "f1", "exact",
           "recall_parent", "recall_child", "recall_spouse", "outside_selected")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", nargs="+", default=["cc_lt_ref_*"],
                        help="table names or shell patterns, matched in both table folders")
    parser.add_argument("--models", nargs="*", default=["tabpfn_v3"], help="models evaluated as detectors (none: classical only)")
    parser.add_argument("--pfm_exps", nargs="+", default=["chain_randomize_local", "chain_noise_local"],
                        help="the column-only perturbations a model is evaluated with")
    parser.add_argument("--metric", default="ccos", help="similarity of the perturbed embedding to the clean one")
    parser.add_argument("--layers", nargs="+", default=["final"],
                        help="what a model is read at: final (its last layer), output (its predicted class "
                             "probabilities, compared by total variation), or a layer name; one detector each")
    parser.add_argument("--probe", nargs="*", default=None, choices=["default", "noise", "hsic"],
                        help="also the embedding variants of detectors/pfm_probe.py, for every model: default "
                             "(perturbations, seeds, readouts, shadow-column thresholds), noise (the noise "
                             "hyperparameters: strength, distribution, rows), hsic (HSIC against dCKA on the best "
                             "configurations); --probe alone is default")
    parser.add_argument("--n_shadows", type=int, default=6, help="shadow columns added to a table by --probe")
    parser.add_argument("--source", default="run", choices=["run", "results"],
                        help="run the models here, or read the results of run_exps.py --exp blanket")
    parser.add_argument("--classical", nargs="*", default=CLASSICAL, help="<detector>:<test>, see detectors/")
    parser.add_argument("--alpha", type=float, default=0.05, help="level of the classical tests")
    parser.add_argument("--truths", nargs="+", default=["graph", "effective"], choices=["graph", "effective"])
    parser.add_argument("--results_dir", default=os.path.join(HERE, "results"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default=None, help="name of the outputs, default: from the first pattern")
    parser.add_argument("--n_jobs", type=int, default=1, help="tables run in parallel by the classical detectors")
    return parser.parse_args()


def find_tables(patterns):
    """The stored tables (simulated or Causal Chamber) whose name matches one of the patterns."""
    names = set()
    for data_dir in (custom_scm_data_dir(), causal_chamber_data_dir()):
        if os.path.isdir(data_dir):
            names.update(f[: -len(".json")] for f in os.listdir(data_dir) if f.endswith(".json"))
    return sorted(n for n in names if any(fnmatch.fnmatch(n, p) for p in patterns))


def column_roles(dataset, truth):
    """The role of the node behind every feature column, in the published or the effective graph
    (a simulated table has only the first: its graph is the one the data was drawn from)."""
    metadata = dataset.metadata
    parents = metadata["effective_parents"] if truth == "effective" and "effective_parents" in metadata \
        else graph_of(metadata)["parents"]
    roles = node_roles(parents, dataset.target_node)
    return np.array([roles[node] for node in dataset.feature_nodes])


def auc(positive, negative):
    """P(a score of `positive` is above a score of `negative`), ties counting half."""
    if len(positive) == 0 or len(negative) == 0:
        return np.nan
    diff = np.subtract.outer(positive, negative)
    return float(((diff > 0) + 0.5 * (diff == 0)).mean())


def evaluate(scores, selected, roles):
    """The ranking and set metrics of one detector on one table, see METRICS."""
    blanket, outside = np.isin(roles, BLANKET), roles == "outside"
    k = int(blanket.sum())
    top = np.argsort(-scores, kind="stable")[:k]
    n_true = int((selected & blanket).sum())
    out = {
        "auc": auc(scores[blanket], scores[outside]),
        "topk_precision": float(blanket[top].mean()) if k else np.nan,
        "topk_exact": float(blanket[top].all()) if k else np.nan,
        "precision": n_true / selected.sum() if selected.any() else np.nan,
        "recall": n_true / k if k else np.nan,
        "f1": 2 * n_true / (selected.sum() + k) if selected.sum() + k else np.nan,
        "exact": float(np.array_equal(selected, blanket)),
        "outside_selected": float(selected[outside].mean()) if outside.any() else np.nan,
    }
    for role in BLANKET:
        out[f"recall_{role}"] = float(selected[roles == role].mean()) if (roles == role).any() else np.nan
    return out


def score_rows(dataset, detector, scores, selected, truths):
    """One row per truth for a detector's answer on a table."""
    target = dataset.metadata.get("target_variable", str(dataset.target_node))
    return [{"table": dataset.name, "target": target, "detector": detector.name, "truth": truth,
             **evaluate(scores, selected, column_roles(dataset, truth))} for truth in truths]


def classical_on_table(name, specs, alpha, truths):
    """The classical detectors on one table (in a worker of its own: they only need the CPU)."""
    dataset = dataset_importer(name)
    rows = []
    for spec in specs:
        detector_name, test = spec.split(":")
        detector = detector_importer(detector_name, {"test": test, "alpha": alpha})
        rows += score_rows(dataset, detector, detector.scores(dataset), detector.select(dataset), truths)
    return rows


def summarise(rows, keys):
    """Mean and standard error of every metric over the tables, by `keys`."""
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, group in groups.items():
        entry = dict(zip(keys, key), n_tables=len(group))
        for metric in METRICS:
            values = np.array([g[metric] for g in group], dtype=float)
            values = values[~np.isnan(values)]
            entry[metric] = float(values.mean()) if len(values) else np.nan
            entry[f"{metric}_se"] = float(values.std() / np.sqrt(len(values))) if len(values) > 1 else np.nan
        out.append(entry)
    return out


def write_csv(rows, path):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def leaderboard(summary, truths, n_tables, tag):
    """The leaderboard as markdown: one table per truth, the detectors sorted by AUC."""
    lines = [f"# Markov blanket identification: {tag}", "",
             f"{n_tables} tables, mean over tables (± s.e.). AUC and top-k rank the columns, the other "
             "columns score the selected set. Recall by role: share of the parents / children / "
             "spouses selected; outside: share of the columns outside the blanket selected.", ""]
    for truth in reversed(truths):  # the effective truth first when there is one
        lines += [f"## {truth} truth", "",
                  "| detector | AUC | top-k prec | precision | recall | F1 | exact | parents | children | spouses | outside |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in sorted((s for s in summary if s["truth"] == truth), key=lambda s: -np.nan_to_num(s["auc"], nan=-1)):
            cells = [f"{s['auc']:.3f} ± {np.nan_to_num(s['auc_se']):.3f}"] + [
                "–" if np.isnan(s[m]) else f"{s[m]:.2f}"
                for m in ("topk_precision", "precision", "recall", "f1", "exact", "recall_parent",
                          "recall_child", "recall_spouse", "outside_selected")]
            lines.append(f"| {s['detector']} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    tables = find_tables(args.tables)
    if not tables:
        raise SystemExit(f"No stored table matches {args.tables} in {custom_scm_data_dir()} or {causal_chamber_data_dir()}")
    tag = args.tag or args.tables[0].replace("*", "").replace("?", "").strip("_") or "tables"
    truths = list(args.truths)
    print(f"{len(tables)} tables: {tables[0]} ... {tables[-1]}")

    # the classical detectors first, in parallel over the tables, before any model puts torch
    # and CUDA in this process
    rows = []
    if args.classical:
        print(f"Classical detectors on {args.n_jobs} cores: {args.classical}")
        for table_rows in Parallel(n_jobs=args.n_jobs)(
            delayed(classical_on_table)(name, args.classical, args.alpha, truths) for name in tables
        ):
            rows += table_rows

    # then every model, loaded once for all the tables
    datasets = [dataset_importer(name) for name in tables]
    for model in args.models:
        print(f"Model {model} on {len(tables)} tables ({args.source})")
        detectors = [detector_importer("pfm", {"model": model, "exp": exp, "metric": args.metric, "layer": layer,
                                               "source": args.source, "results_dir": args.results_dir,
                                               "device": args.device})
                     for layer in args.layers for exp in args.pfm_exps]
        if args.probe is not None:
            for grid_name in args.probe or ["default"]:
                detectors += probe_grid(model, args.n_shadows, args.results_dir, args.device, grid_name)
        for dataset in datasets:
            for detector in detectors:
                try:
                    scores = detector.scores(dataset)
                except (FileNotFoundError, KeyError) as error:  # --source results: not run on this table
                    print(f"  skipped: {error}")
                    continue
                rows += score_rows(dataset, detector, scores, detector.select(dataset), truths)
        release_models()
    if not rows:
        raise SystemExit("No detector could be evaluated")

    out_dir = os.path.join(args.results_dir, "benchmark")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"blanket_{tag}")
    truths = [t for t in truths if any(r["truth"] == t for r in rows)]
    summary = summarise(rows, ("detector", "truth"))
    write_csv(rows, f"{base}.csv")
    write_csv(summary, f"{base}_summary.csv")
    write_csv(summarise(rows, ("target", "detector", "truth")), f"{base}_by_target.csv")
    with open(f"{base}_tables.json", "w") as file:
        json.dump(tables, file)
    board = leaderboard(summary, truths, len(tables), tag)
    with open(f"{base}_leaderboard.md", "w") as file:
        file.write(board + "\n")
    print(f"Wrote {base}_leaderboard.md\n\n{board}")


if __name__ == "__main__":
    main()
