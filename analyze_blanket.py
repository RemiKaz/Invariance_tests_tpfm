"""Does a model's embedding encode the Markov blanket of the target?

Reads the results of the causal experiments (never runs a model):

- `chain_all` on the DAG tables (dagpool_*): one sensitivity per perturbed node, 1 - similarity to
  the baseline. In the `*_local` variants one column changes whatever the node is, so the node's
  role is the only thing that differs between points. If a layer encodes what P(y | X) depends
  on, the nodes of the blanket (parents, children, spouses) should move it more than the nodes
  outside it, which a perfect P(y | X) ignores once the blanket is observed.
- `chain_local` on the chains: the target's parent is its whole blanket.

Two statistics per layer, for every table then averaged (the error is the standard error over
the tables):

- AUC: the probability that a blanket node moves the embedding more than an outside node
  (0.5 = no selectivity, 1 = every blanket node more than every outside node).
- Controlled: the same, after removing what a node's marginal association with y explains. The
  nodes outside the blanket include the ancestors of the parents, which correlate with y without
  being needed to predict it, so a layer that merely follows "correlated with y" would score
  above 0.5 on the AUC. The coefficient is the one of the blanket indicator in a regression of
  the within-table rank of the sensitivity on the indicator and on the within-table rank of
  |Spearman(column, y)| (0 = nothing beyond the association).

    python analyze_blanket.py --model limix_v2

Writes `<results_dir>/_fused/<model>_blanket_selectivity.csv` and `.png`.
"""

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

from dataset_generator.scm_graph import graph_of, node_roles
from datasets.custom_scm_dataset import custom_scm_paths
from utils.metrics import metric_label
from utils.results import layer_key, load, run_path
from utils.style import BACKGROUND, BLUE_RAMP, INK, MUTED, ORANGE_RAMP, style_axes

HERE = os.path.dirname(os.path.abspath(__file__))
BLANKET = ("parent", "child", "spouse")
LOCAL = ("chain_noise_local", "chain_randomize_local")


def auc(positive, negative):
    """P(a value of `positive` is above a value of `negative`), ties counting half."""
    diff = np.subtract.outer(positive, negative)
    return float(((diff > 0) + 0.5 * (diff == 0)).mean())


def within_rank(values):
    """The ranks of `values` scaled to [0, 1], 0.5 when they are all equal."""
    return (rankdata(values) - 1) / max(len(values) - 1, 1)


def sensitivity(payload, layer, metric, exp):
    """1 - similarity per perturbed node (the level is the node), or None if not available."""
    for result in payload["layers"].get(layer, {}).get(metric, []):
        if result["name"] == exp:
            return 1 - np.array([np.nan if s is None else s for s in result["similarities"]], dtype=float)
    return None


def load_tables(model, results_dir, prefix="dagpool_"):
    """Every DAG table with a chain_all result: (name, payload, roles, |Spearman(node, y)|)."""
    tables = []
    for name in sorted(os.listdir(os.path.join(results_dir, "runs", model))):
        payload = load(run_path(results_dir, model, name, "chain_all"))
        if not name.startswith(prefix) or payload is None:
            continue
        with open(custom_scm_paths(name)[1]) as file:
            metadata = json.load(file)
        roles = node_roles(graph_of(metadata)["parents"], metadata["target_node"])
        table = np.load(custom_scm_paths(name)[0])
        association = np.array([abs(spearmanr(table["nodes"][:, k], table["y"])[0]) for k in range(len(roles))])
        tables.append((name, payload, roles, np.nan_to_num(association)))
    return tables


def table_statistics(sens, roles, association):
    """AUCs (blanket, parent, child, spouse vs outside), the controlled coefficient and the
    sensitivity of the target (its labels are the ones that change) for one table."""
    roles = np.array(roles)
    keep = (roles != "target") & ~np.isnan(sens)
    outside = sens[keep & (roles == "outside")]
    out = {"blanket": auc(sens[keep & np.isin(roles, BLANKET)], outside)}
    for role in BLANKET:
        mask = keep & (roles == role)
        out[role] = auc(sens[mask], outside) if mask.any() else np.nan
    # regression of the within-table sensitivity rank on the blanket indicator, controlling for
    # the rank of the association with y (both centred in the table: a table fixed effect)
    y = within_rank(sens[keep])
    blanket = np.isin(roles[keep], BLANKET).astype(float)
    rank_assoc = within_rank(association[keep])
    X = np.column_stack([blanket - blanket.mean(), rank_assoc - rank_assoc.mean()])
    out["controlled_design"] = (X, y - y.mean())
    target = sens[roles == "target"]
    out["target"] = float(target[0]) if len(target) and not np.isnan(target[0]) else np.nan
    out["blanket_mean"] = float(np.nanmean(sens[keep & np.isin(roles, BLANKET)]))
    return out


def controlled_coefficient(designs):
    """The blanket coefficient of the pooled within-table regression."""
    X = np.vstack([x for x, _ in designs])
    y = np.concatenate([y for _, y in designs])
    return float(np.linalg.lstsq(X, y, rcond=None)[0][0])


def summarise(tables, layers, metric, exp, n_boot=300, seed=0):
    """layer -> statistics averaged over the tables."""
    rng = np.random.default_rng(seed)
    rows = {}
    for layer in layers:
        stats = []
        for _, payload, roles, association in tables:
            sens = sensitivity(payload, layer, metric, exp)
            if sens is not None:
                stats.append(table_statistics(sens, roles, association))
        if not stats:
            continue
        row = {"n_tables": len(stats)}
        for key in ("blanket", "parent", "child", "spouse"):
            values = np.array([s[key] for s in stats])
            row[key] = float(np.nanmean(values))
            row[f"{key}_se"] = float(np.nanstd(values) / np.sqrt(np.sum(~np.isnan(values))))
        designs = [s["controlled_design"] for s in stats]
        row["controlled"] = controlled_coefficient(designs)
        boot = [controlled_coefficient([designs[i] for i in rng.integers(0, len(designs), len(designs))]) for _ in range(n_boot)]
        row["controlled_se"] = float(np.std(boot))
        row["target_over_blanket"] = float(np.nanmean([s["target"] / s["blanket_mean"] for s in stats]))
        rows[layer] = row
    return rows


def chain_statistics(model, results_dir):
    """For each chain table, the sensitivity of every node and the rank of its distance to the
    target (the parent, the whole blanket, is the node just before it)."""
    out = {}
    runs = os.path.join(results_dir, "runs", model)
    for name in sorted(os.listdir(runs)):
        payload = load(run_path(results_dir, model, name, "chain_local"))
        if payload is None or not name.startswith("chain_"):
            continue
        out[name] = payload
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="limix_v2")
    parser.add_argument("--metrics", nargs="+", default=["ccos", "nmi"])
    parser.add_argument("--results_dir", default=os.path.join(HERE, "results"))
    args = parser.parse_args()

    tables = load_tables(args.model, args.results_dir)
    print(f"{len(tables)} DAG tables with a chain_all result for {args.model}")
    layers = sorted(tables[0][1]["layers"], key=layer_key)

    out_dir = os.path.join(args.results_dir, "_fused")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{args.model}_blanket_selectivity.csv")
    summary = {}
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "experiment", "layer", "n_tables", "auc_blanket", "auc_blanket_se", "auc_parent",
                         "auc_child", "auc_spouse", "controlled", "controlled_se", "target_over_blanket"])
        for metric in args.metrics:
            for exp in LOCAL:
                rows = summarise(tables, layers, metric, exp)
                summary[metric, exp] = rows
                for layer, r in rows.items():
                    writer.writerow([metric, exp, layer, r["n_tables"], r["blanket"], r["blanket_se"], r["parent"],
                                     r["child"], r["spouse"], r["controlled"], r["controlled_se"], r["target_over_blanket"]])

    fig, axes = plt.subplots(2, len(args.metrics), figsize=(6.5 * len(args.metrics), 7.5), facecolor=BACKGROUND,
                             squeeze=False, sharex=True)
    colors = {LOCAL[0]: BLUE_RAMP[4], LOCAL[1]: ORANGE_RAMP[3]}
    for j, metric in enumerate(args.metrics):
        for exp in LOCAL:
            rows = summary[metric, exp]
            x = np.arange(len(rows))
            for i, (key, ref) in enumerate((("blanket", 0.5), ("controlled", 0.0))):
                ax = axes[i, j]
                style_axes(ax)
                y = np.array([r[key] for r in rows.values()])
                se = np.array([r[f"{key}_se"] for r in rows.values()])
                ax.fill_between(x, y - se, y + se, color=colors[exp], alpha=0.18, linewidth=0)
                ax.plot(x, y, color=colors[exp], marker="o", markersize=3.5, linewidth=1.6, label=exp.replace("chain_", ""))
                ax.axhline(ref, color=MUTED, linewidth=1.0, linestyle="--")
                ax.set_xticks(x)
                ax.set_xticklabels(list(rows), rotation=90, fontsize=7, color=MUTED)
        axes[0, j].set_title(metric_label(metric), color=INK, fontsize=11)
        axes[0, j].set_ylabel("AUC: blanket node moves more than an outside node", color=MUTED, fontsize=9)
        axes[1, j].set_ylabel("blanket effect beyond association with y", color=MUTED, fontsize=9)
        axes[1, j].set_xlabel("layer", color=MUTED, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(f"{args.model}: does the embedding follow the Markov blanket? ({len(tables)} DAGs, column-only perturbations)",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{args.model}_blanket_selectivity.png"), dpi=150, facecolor=BACKGROUND)
    plt.close(fig)

    for (metric, exp), rows in summary.items():
        print(f"\n== {metric} / {exp}: AUC blanket vs outside (parent / child / spouse), controlled coefficient, target / blanket")
        for layer, r in rows.items():
            print(f"  {layer:>5s}  AUC {r['blanket']:.2f}±{r['blanket_se']:.2f} ({r['parent']:.2f} / {r['child']:.2f} / {r['spouse']:.2f})"
                  f"  ctrl {r['controlled']:+.2f}±{r['controlled_se']:.2f}  target/blanket {r['target_over_blanket']:.1f}")

    chains = chain_statistics(args.model, args.results_dir)
    for metric in args.metrics:
        print(f"\n== chains, {metric}, chain_randomize_local: sensitivity of the parent (the blanket) / of the other features, by layer")
        for layer in layers[::3] + [layers[-1]]:
            parts = []
            for name, payload in chains.items():
                sens = sensitivity(payload, layer, metric, "chain_randomize_local")
                if sens is not None:
                    parts.append((sens[-2], np.nanmean(sens[:-2])))
            if parts:
                print(f"  {layer:>5s}  " + "   ".join(f"{p:.2f}/{o:.2f}" for p, o in parts))
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
