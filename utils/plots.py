"""The fused figures of fuse_plot.py: one panel per experiment, one curve per layer (or per role
of the perturbed node), and the layout they share with compare_groups.py."""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb

from dataset_generator.scm_graph import ROLES
from utils.metrics import metric_label
from utils.style import AQUA_RAMP, BACKGROUND, BLUE_RAMP, INK, MARKERS, MUTED, ORANGE_RAMP, ROLE_STYLE, SECONDARY, style_axes

MODULE_RAMPS = {"col": BLUE_RAMP, "row": ORANGE_RAMP, "icl": AQUA_RAMP}
# Models made of a column, a row and an ICL module: one hue per module, darker with depth
# inside it. (TabFM has five modules and keeps the single ramp.)
MODULAR_MODELS = ("tabicl_v1", "tabicl_v2", "tabpfn_v3")
MAX_DISCRETE_LAYERS = len(BLUE_RAMP)
HEADER_INCHES = 0.65  # room for the column headers above the first panel titles

# The grid has one column per side of the table that is perturbed. Within a column the panels
# follow these orders, so that the two sides line up (shuffle next to shuffle, remove next to
# remove, y next to the alternative target); an unknown experiment goes at the end of the
# rows column.
ROW_PERTURBATIONS = ("row_shuffle", "row_remove", "random_labels", "value_noise")
COLUMN_PERTURBATIONS = ("col_shuffle", "col_remove", "alt_target")
CHAIN_PERTURBATIONS = ("chain_noise", "chain_randomize", "chain_edge_shuffle")
CONTEXT_PERTURBATIONS = ("context_swap",)


def panel_columns(names):
    """Split experiment names into the grid columns, as (header, names) pairs.

    Row perturbations (shuffle/remove train rows, random labels, random X values) go in the
    first column, column perturbations (shuffle/remove columns, alternative target) in the
    second, causal perturbations of a custom SCM dataset in the third (propagated to the
    descendants) and fourth (the one column only); an empty column is left out.
    """

    def in_order(group, order):
        return sorted(group, key=lambda name: order.index(name) if name in order else len(order))

    chain = in_order([n for n in names if n in CHAIN_PERTURBATIONS], CHAIN_PERTURBATIONS)
    local = in_order(
        [n for n in names if n.endswith("_local") and n[: -len("_local")] in CHAIN_PERTURBATIONS],
        [f"{n}_local" for n in CHAIN_PERTURBATIONS],
    )
    columns = in_order([n for n in names if n in COLUMN_PERTURBATIONS], COLUMN_PERTURBATIONS)
    context = [n for n in names if n in CONTEXT_PERTURBATIONS]
    placed = set(chain + local + columns + context)
    rows = in_order([n for n in names if n not in placed], ROW_PERTURBATIONS)
    pairs = (
        ("Row perturbations", rows),
        ("Column perturbations", columns),
        ("Causal perturbations", chain),
        ("Causal perturbations, column only", local),
        ("Context from another table", context),
    )
    return [(header, group) for header, group in pairs if group]


def finish_layout(fig, axes, headers, top_element=None, gap_inches=0.3):
    """Lay the panels out under `top_element`, with a header above each column.

    tight_layout first, for good spacing between the panels; only then is the top pulled down
    to clear `top_element` (title, legend...) and the headers, measured on the rendered
    figure. tight_layout(rect=...) can't do it, its own padding doesn't shrink to a requested
    value, subplots_adjust(top=...) takes the figure fraction literally.

    Args:
        fig: The figure.
        axes: The 2-D array of axes, one column per header.
        headers (list): The column headers.
        top_element (optional): The artist above the panels. Defaults to None (top of figure).
        gap_inches (float, optional): The space between top_element and the headers.
    """
    fig.tight_layout()
    fig.canvas.draw()
    top = 1.0
    if top_element is not None:
        extent = top_element.get_tightbbox(fig.canvas.get_renderer())  # with its ticks and title
        top = extent.transformed(fig.transFigure.inverted()).y0
    top -= gap_inches / fig.get_figheight()
    fig.subplots_adjust(top=top - HEADER_INCHES / fig.get_figheight())
    for j, header in enumerate(headers):
        box = axes[0, j].get_position()
        fig.text(
            (box.x0 + box.x1) / 2, top, header, color=INK, fontsize=12, fontweight="semibold",
            ha="center", va="top",
        )


def layer_module(layer):
    """The module of a layer of a TabICL-like model: "col", "row" or "icl", else None."""
    if layer.startswith(("col_", "tf_col")):
        return "col"
    if layer.startswith(("row_", "tf_row")):
        return "row"
    if layer.startswith("icl_") or layer.lstrip("-").isdigit() or layer in ("tf_icl", "last"):
        return "icl"
    return None


def module_colors(layers):
    """Colors of the layers, one hue per module and a gradient inside each of them.

    Returns:
        tuple: the color of every layer, and the runs of layers of one module as
            (module, indices) pairs; or None if a layer belongs to no module.
    """
    modules = [layer_module(layer) for layer in layers]
    if None in modules:
        return None
    inside = {m: layer_colors(modules.count(m), MODULE_RAMPS[m]) for m in set(modules)}
    seen = dict.fromkeys(inside, 0)
    colors, groups = [], []
    for i, module in enumerate(modules):
        colors.append(inside[module][seen[module]])
        seen[module] += 1
        if groups and groups[-1][0] == module:
            groups[-1][1].append(i)
        else:
            groups.append((module, [i]))
    return colors, groups


def layer_colors(n_layers, ramp=BLUE_RAMP):
    """One color per layer, darker with depth.

    Evenly-spaced discrete steps of the ramp up to its own length (10): past 5 the result no
    longer clears the ordinal contrast floor on its own, which is why layers also get a
    distinct marker shape - see MARKERS. Beyond the ramp's own length, falls back to
    interpolating between its stops.
    """
    if n_layers == 1:
        return [ramp[4]]
    if n_layers <= len(ramp):
        return [ramp[round(i * (len(ramp) - 1) / (n_layers - 1))] for i in range(n_layers)]
    stops = [to_rgb(color) for color in ramp]
    colors = []
    for i in range(n_layers):
        position = i * (len(stops) - 1) / (n_layers - 1)
        low = int(position)
        high = min(low + 1, len(stops) - 1)
        fraction = position - low
        colors.append(tuple(a + (b - a) * fraction for a, b in zip(stops[low], stops[high])))
    return colors


def _values(result):
    """The similarities of a result, None (an undefined level) read as NaN: a gap in the curve."""
    return [math.nan if value is None else value for value in result["similarities"]]


def _lowest(values):
    defined = [value for value in values if value == value]
    return min(defined) if defined else math.inf


def _experiments(runs, layers):
    """experiment name -> (title, xlabel), in the order they were run."""
    experiments = {}
    for layer in layers:
        for result in runs[layer]:
            experiments.setdefault(result["name"], (result["title"], result["xlabel"]))
    return experiments


def _set_ylim(ax, lowest, full_scale):
    """The curves start at 1 and the layers differ by a few percent: zoom on the range they
    cover, on a round bound, unless the full scale is asked for. dcka can dip slightly below 0
    and cosine down to -1, so the bottom follows the data."""
    bottom = math.floor((lowest - 0.02) * 20) / 20 if lowest < math.inf else 0.95
    if full_scale:
        ax.set_ylim(min(0.0, bottom), 1.05)
    else:
        ax.set_ylim(min(0.95, bottom), 1.02)


def _save(fig, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def plot_layers(runs, layers, title, out_path, metric, full_scale=False, modular=False):
    """One panel per experiment, one curve per layer.

    Args:
        runs (dict): layer -> list of experiment results.
        layers (list): The layers to draw, in this order.
        title (str): The title of the figure.
        out_path (str): The path of the png.
        metric (str): The metric of the results, for the axis label.
        full_scale (bool, optional): Draw the y axis from 0 to 1. Defaults to False.
        modular (bool, optional): One hue per module (col, row, icl) when every layer belongs
            to one. Defaults to False.
    """
    colors, modules = layer_colors(len(layers)), None
    if modular and module_colors(layers):
        colors, modules = module_colors(layers)
    experiments = _experiments(runs, layers)
    columns = panel_columns(experiments)  # row perturbations left, column perturbations right
    n_cols, n_rows = len(columns), max(len(group) for _, group in columns)
    # A legend past a handful of entries wraps into extra rows, needing a little more headroom;
    # the exact top margin is measured after the legend is drawn, in finish_layout.
    with_markers = modules is None and len(layers) <= MAX_DISCRETE_LAYERS
    legend_cols = min(len(layers), 4) if with_markers else 0
    legend_rows = -(-len(layers) // legend_cols) if legend_cols else 0
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(5 * n_cols, 8), 4 * n_rows + HEADER_INCHES + 0.8 + 0.3 * max(0, legend_rows - 1)),
        facecolor=BACKGROUND,
        squeeze=False,
        sharey=True,
    )

    markers = [MARKERS[i % len(MARKERS)] for i in range(len(layers))] if with_markers else [None] * len(layers)
    lowest = math.inf
    for j, (_, group) in enumerate(columns):
        for i, name in enumerate(group):
            ax = axes[i, j]
            panel_title, xlabel = experiments[name]
            style_axes(ax)
            for layer, color, marker in zip(layers, colors, markers):
                result = next((r for r in runs[layer] if r["name"] == name), None)
                if result is None:
                    continue
                lowest = min(lowest, _lowest(_values(result)))
                ax.plot(
                    result["levels"],
                    _values(result),
                    color=color,
                    linewidth=1.6,
                    marker=marker,
                    markersize=5.5 if marker in ("*", "P", "X") else 4.5,  # visually smaller shapes
                    markeredgecolor=BACKGROUND,  # a gap of surface where the marks overlap
                    markeredgewidth=1.0,
                )
            ax.set_title(panel_title, color=INK, fontsize=11)
            ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
        for ax in axes[len(group) :, j]:
            ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(metric_label(metric), color=MUTED, fontsize=9)
    _set_ylim(axes.flat[0], lowest, full_scale)

    suptitle = fig.suptitle(title, color=INK, fontsize=13, y=0.985)
    if with_markers:
        # a legend, in the order of the depth, colors and shapes matching the lines. Anchored
        # just under the title and left to grow downward as rows are added.
        handles = [
            plt.Line2D([], [], color=color, linewidth=1.6, marker=marker, markersize=5.5)
            for color, marker in zip(colors, markers)
        ]
        legend = fig.legend(
            handles, layers, title="layer", loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=legend_cols,
            frameon=False, labelcolor=SECONDARY, fontsize=9, title_fontsize=9,
        )
        legend.get_title().set_color(MUTED)
        top_element = legend
    else:
        # too many layers for color + shape alone: a depth scale, one cell per layer
        cax = fig.add_axes((0.2, 0.915, 0.6, 0.016))
        scale = ScalarMappable(norm=BoundaryNorm(range(len(layers) + 1), len(layers)), cmap=ListedColormap(colors))
        if modules:  # one label per module, centered under its cells
            ticks = [(indices[0] + indices[-1] + 1) / 2 for _, indices in modules]
            labels = [module for module, _ in modules]
        else:
            ticks, labels = [i + 0.5 for i in range(len(layers))], layers
        bar = fig.colorbar(scale, cax=cax, orientation="horizontal", ticks=ticks)
        bar.ax.set_xticklabels(labels)
        bar.ax.tick_params(colors=SECONDARY, labelsize=8, length=0)
        bar.outline.set_visible(False)
        bar.minorticks_off()
        note = ", one color per module" if modules else ""
        bar.ax.set_title(f"layer (shallow to deep{note})", color=MUTED, fontsize=9, pad=4)
        top_element = bar.ax

    finish_layout(fig, axes, [header for header, _ in columns], top_element)
    suptitle.set_y(0.985)  # subplots_adjust can nudge the suptitle along with the rest back down
    _save(fig, out_path)


def perturbed_node(name, level):
    """The node an experiment perturbs at a level: the node itself, or for the experiments that
    shuffle the edges into a node, the child of the edge (level k is node k+1)."""
    return level + 1 if "edge_shuffle" in name else level


def plot_roles(runs, layers, roles, title, out_path, metric, full_scale=False):
    """One panel per experiment, the layers on the x axis and one curve per role of the perturbed
    node: the mean over the nodes of that role. It shows which layers react to the nodes of the
    Markov blanket of the target, and which ones to the nodes outside it."""
    experiments = _experiments(runs, layers)
    columns = panel_columns(experiments)
    n_cols, n_rows = len(columns), max(len(group) for _, group in columns)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(max(6.5 * n_cols, 8), 3.8 * n_rows + HEADER_INCHES + 1.0),
        facecolor=BACKGROUND, squeeze=False, sharey=True,
    )
    x = list(range(len(layers)))
    lowest, present = math.inf, {}
    for j, (_, group) in enumerate(columns):
        for i, name in enumerate(group):
            ax = axes[i, j]
            style_axes(ax)
            by_layer = []
            for layer in layers:
                result = next((r for r in runs[layer] if r["name"] == name), None)
                by_layer.append(
                    {perturbed_node(name, lv): s for lv, s in zip(result["levels"], result["similarities"]) if s is not None}
                    if result else {}
                )
            for role in ROLES:
                nodes = [n for n, r in enumerate(roles) if r == role]
                curve = [
                    sum(v[n] for n in nodes if n in v) / sum(n in v for n in nodes) if any(n in v for n in nodes) else math.nan
                    for v in by_layer
                ]
                if all(math.isnan(c) for c in curve):
                    continue
                present[role] = len([n for n in nodes if any(n in v for v in by_layer)])
                color, marker = ROLE_STYLE[role]
                lowest = min(lowest, _lowest(curve))
                ax.plot(x, curve, color=color, marker=marker, markersize=4.5, linewidth=1.6,
                        markeredgecolor=BACKGROUND, markeredgewidth=1.0)
            ax.set_title(experiments[name][0], color=INK, fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(layers, rotation=90, fontsize=7, color=MUTED)
            ax.tick_params(axis="x", labelbottom=True, length=2)
        for ax in axes[len(group):, j]:
            ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(metric_label(metric), color=MUTED, fontsize=9)
    _set_ylim(axes.flat[0], lowest, full_scale)

    suptitle = fig.suptitle(title, color=INK, fontsize=13, y=0.985)
    shown = [r for r in ROLES if r in present]
    handles = [plt.Line2D([], [], color=ROLE_STYLE[r][0], marker=ROLE_STYLE[r][1], linewidth=1.6, markersize=6) for r in shown]
    legend = fig.legend(handles, [f"{r} ({present[r]})" for r in shown], title="role of the perturbed node (number of nodes)",
                        loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=len(handles), frameon=False,
                        labelcolor=SECONDARY, fontsize=9, title_fontsize=9)
    legend.get_title().set_color(MUTED)
    finish_layout(fig, axes, [header for header, _ in columns], legend)
    suptitle.set_y(0.985)
    _save(fig, out_path)
