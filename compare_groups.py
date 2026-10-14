"""Compare, over the groups of tables, how the embeddings move when the context is swapped.

Every group (dataset_generator/cond_groups.py) is run with the experiment `context_swap` on its
baseline table: the model reads the same query rows with the context taken from each of the four
other tables, and the embeddings are compared to the ones read with the baseline's own context.
This script puts the groups together: for each layer and each table, the mean similarity over the
groups and the range between the groups.

    python run_exps.py --model mitra_v2 --dataset cg00_baseline --exp context_swap --layer 0 last
    python compare_groups.py --model mitra_v2 --metric ccos

The figure is written to `<results_dir>/_fused/<model>_context_swap_<metric>_groups.png`, and with
--csv the plotted numbers next to it.
"""

import argparse
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset_generator.cond_groups import DESCRIPTIONS, VARIANTS
from utils.metrics import metric_label
from utils.results import find_combinations, layer_key, layers_of, load, run_path
from utils.style import BACKGROUND, GRID_COLOR, INK, MUTED, SECONDARY, style_axes

DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Colour and marker by what the table shares with the baseline: the two that keep the whole
# joint distribution are cool colours, the one that keeps P(y | X) only is orange, the one that
# changes P(y | X) is ink.
STYLE = {
    "markov_equiv": ("#2a78d6", "o"),
    "anticausal": ("#1baf7a", "s"),
    "same_cond": ("#eb6834", "^"),
    "near_graph": (INK, "v"),
}
LABELS = {
    "markov_equiv": "markov_equiv\nsame joint",
    "anticausal": "anticausal\nsame joint",
    "same_cond": "same_cond\nsame P(y|X), other P(X)",
    "near_graph": "near_graph\nother P(y|X)",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model of the runs, e.g. mitra_v2")
    parser.add_argument("--metric", default="ccos", help="Metric of the runs (default ccos)")
    parser.add_argument("--exp", default="context_swap", help="Experiment of the runs")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--csv", action="store_true", help="Also write the plotted numbers as a csv")
    return parser.parse_args()


def load_groups(results_dir, model, exp, metric):
    """The similarity of every group, layer and table to the baseline's own embeddings.

    Returns:
        tuple: the datasets (the baseline of each group), the layers by depth, and an array
            (groups, layers, tables) in VARIANTS order, NaN where a run is missing.
    """
    datasets = sorted(d for m, d, e, me in find_combinations(results_dir, exp, metric) if m == model)
    if not datasets:
        raise SystemExit(f"No result for model={model} exp={exp} metric={metric} in {results_dir}")
    runs = {dataset: layers_of(load(run_path(results_dir, model, dataset, exp)), metric) for dataset in datasets}
    layers = sorted({layer for r in runs.values() for layer in r}, key=layer_key)
    values = np.full((len(datasets), len(layers), len(VARIANTS)), np.nan)
    for i, dataset in enumerate(datasets):
        for j, layer in enumerate(layers):
            for result in runs[dataset].get(layer, []):
                if result["name"] == exp and len(result["similarities"]) == len(VARIANTS):
                    values[i, j] = result["similarities"]
    return datasets, layers, values


def plot_groups(model, metric, datasets, layers, values, out_path):
    """One line per table: the mean over the groups, with the range between the groups."""
    mean, low, high = (f(values, axis=0) for f in (np.nanmean, np.nanmin, np.nanmax))
    x = np.arange(len(layers))
    shown = [k for k, variant in enumerate(VARIANTS) if variant != "baseline"]
    fig, ax = plt.subplots(figsize=(max(8.5, 0.32 * len(layers) + 4.5), 5.6), facecolor=BACKGROUND)
    style_axes(ax, grid_axis="y")
    for k in shown:
        color, marker = STYLE[VARIANTS[k]]
        ax.fill_between(x, low[:, k], high[:, k], color=color, alpha=0.13, linewidth=0)
        ax.plot(x, mean[:, k], color=color, marker=marker, markersize=4.5, linewidth=1.8,
                markeredgecolor=BACKGROUND, markeredgewidth=0.8)
    ax.axhline(1.0, color=GRID_COLOR, linewidth=1.0)
    ax.set_ylim(min(0.95, math.floor((np.nanmin(low[:, shown]) - 0.02) * 20) / 20), 1.02)

    # direct labels in the margin on the right, at the height of the end of each line, pushed
    # apart where the lines end close together (axes coordinates, so they follow the axis range)
    y0, y1 = ax.get_ylim()
    wanted = sorted(((mean[-1, k] - y0) / (y1 - y0), k) for k in shown)[::-1]
    placed, floor = [], 2.0
    for target, k in wanted:
        y = min(target, floor - 0.1)
        placed.append((k, y))
        floor = y
    for k, y in placed:
        ax.annotate(LABELS[VARIANTS[k]], xy=(x[-1], mean[-1, k]), xycoords="data", xytext=(1.02, y),
                    textcoords="axes fraction", fontsize=9, color=INK, va="center", annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": STYLE[VARIANTS[k]][0], "lw": 0.8, "shrinkA": 0, "shrinkB": 3})
    ax.set_xlim(-0.5, len(layers) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=90, fontsize=8, color=MUTED)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlabel("layer", color=MUTED, fontsize=9)
    ax.set_ylabel(metric_label(metric) + " to the baseline's own context", color=MUTED, fontsize=9)
    fig.suptitle(f"{model}: the same query rows read with the context of another table", color=INK, fontsize=13, x=0.06, ha="left")
    fig.text(0.06, 0.915, f"{metric_label(metric)}; mean of {len(datasets)} groups, band = range between groups",
             color=SECONDARY, fontsize=9.5, ha="left")
    fig.subplots_adjust(left=0.08, right=0.72, top=0.87, bottom=0.2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_table(layers, values, out_path):
    """The plotted numbers: layer, table, mean, min and max over the groups, number of groups."""
    with open(out_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["layer", "table", "mean", "min", "max", "n_groups"])
        for j, layer in enumerate(layers):
            for k, variant in enumerate(VARIANTS):
                column = values[:, j, k]
                column = column[~np.isnan(column)]
                if len(column):
                    writer.writerow([layer, variant, column.mean(), column.min(), column.max(), len(column)])


def main():
    args = parse_args()
    datasets, layers, values = load_groups(args.results_dir, args.model, args.exp, args.metric)
    out_path = os.path.join(args.results_dir, "_fused", f"{args.model}_{args.exp}_{args.metric}_groups.png")
    plot_groups(args.model, args.metric, datasets, layers, values, out_path)
    if args.csv:
        write_table(layers, values, os.path.splitext(out_path)[0] + ".csv")
    print(f"{len(datasets)} group(s), {len(layers)} layer(s): {out_path}")
    mean = np.nanmean(values, axis=0)
    print("mean similarity over groups, first / middle / last layer:")
    for k, variant in enumerate(VARIANTS):
        if variant != "baseline":
            print(f"  {variant:13s} {mean[0, k]:.2f} / {mean[len(layers) // 2, k]:.2f} / {mean[-1, k]:.2f}   ({DESCRIPTIONS[variant]})")


if __name__ == "__main__":
    main()
