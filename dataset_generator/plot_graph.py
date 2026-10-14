"""Draw the causal graph of a stored table, its nodes coloured by their role towards the target.

    python -m dataset_generator.plot_graph --name dag_8f_3c_blanket_deg2_relnoise_seed0

The figure is written next to the table, as `<name>_graph.png`.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from dataset_generator.scm_graph import ROLES, graph_of, node_info, node_roles
from datasets.custom_scm_dataset import custom_scm_data_dir, custom_scm_paths
from utils.style import BACKGROUND, INK, MUTED, ROLE_STYLE, SECONDARY

# Colour and shape both say the role, so the figure does not rely on hue alone. The nodes
# outside the blanket are drawn as light circles, behind the ones that matter.
ROLE_COLOR = {role: color for role, (color, _) in ROLE_STYLE.items()} | {"outside": "#c3c2b7"}
ROLE_MARKER = {role: marker for role, (_, marker) in ROLE_STYLE.items()} | {"outside": "o"}
ROLE_LABEL = {
    "target": "target",
    "parent": "parent",
    "child": "child",
    "spouse": "spouse (other parent of a child)",
    "outside": "outside the Markov blanket",
}
LIGHT_TEXT = ("target", "parent")  # roles dark enough for white text


def plot_graph(name, data_dir=None, out_path=None):
    """Draw the graph of a stored table.

    Nodes are placed by depth (longest path from a root), edges go left to right. The edges
    touching the target are thick, the ones inside its Markov blanket are darker than the rest.
    Under each node is the column of X it is (or `y` for the target).

    Args:
        name (str): The name of the stored table.
        data_dir (str, optional): The folder of the tables. Defaults to `custom_scm_data_dir()`.
        out_path (str, optional): Where to write the png. Defaults to `<name>_graph.png` in
            the data folder.

    Returns:
        str: The path of the figure.
    """
    data_dir = data_dir or custom_scm_data_dir()
    with open(custom_scm_paths(name, data_dir)[1]) as file:
        metadata = json.load(file)
    parents = graph_of(metadata)["parents"]
    target = metadata["target_node"]
    roles, depth = node_roles(parents, target), node_info(parents)["depth"]
    column = {node: j for j, node in enumerate(metadata["feature_nodes"])}
    n = len(parents)

    by_depth = {}
    for k in range(n):
        by_depth.setdefault(depth[k], []).append(k)
    pos = {}
    for level, nodes in by_depth.items():
        for i, k in enumerate(nodes):
            pos[k] = (level, 1.15 * ((len(nodes) - 1) / 2 - i))
    n_levels, height = max(depth) + 1, max(len(v) for v in by_depth.values())

    fig, ax = plt.subplots(figsize=(max(7.0, 1.7 * n_levels + 1.5), max(4.2, 1.15 * height + 2.4)), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    for child, ps in enumerate(parents):
        for p in ps:
            inside = "outside" not in (roles[p], roles[child])
            touches_target = target in (p, child)
            span = depth[child] - depth[p]
            rad = min(0.2 * (span - 1), 0.32) * (1 if (p + child) % 2 == 0 else -1)  # long edges bow away
            ax.add_patch(
                FancyArrowPatch(
                    pos[p], pos[child], connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=14,
                    color=INK if touches_target else (SECONDARY if inside else "#b9b8b0"),
                    lw=1.9 if touches_target else 1.1, shrinkA=16, shrinkB=16, zorder=1,
                )
            )
    for k, (x, y) in pos.items():
        role = roles[k]
        ax.scatter([x], [y], s=820, color=ROLE_COLOR[role], marker=ROLE_MARKER[role], edgecolor=BACKGROUND, linewidth=2, zorder=3)
        ax.text(x, y, str(k), ha="center", va="center", fontsize=12, fontweight="bold", zorder=4,
                color="white" if role in LIGHT_TEXT else INK)
        ax.text(x, y - 0.42, "y" if k == target else f"col {column[k]}", ha="center", va="top", fontsize=7.5,
                color=SECONDARY, zorder=4, bbox={"facecolor": BACKGROUND, "edgecolor": "none", "pad": 1.5})

    ys = [y for _, y in pos.values()]
    ax.set_xlim(-0.6, n_levels - 0.4)
    room = 0.22 * max(depth[c] - depth[p] for c, ps in enumerate(parents) for p in ps)  # the bowed edges
    ax.set_ylim(min(ys) - 0.95 - room, max(ys) + 0.75 + room)
    ax.set_xticks(range(n_levels))
    ax.set_xticklabels([f"depth {i}" for i in range(n_levels)], color=MUTED, fontsize=9)
    ax.set_yticks([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    present = [role for role in ROLES if role in roles]
    handles = [
        plt.Line2D([], [], marker=ROLE_MARKER[r], ls="", ms=10, color=ROLE_COLOR[r], label=f"{ROLE_LABEL[r]} ({roles.count(r)})")
        for r in present
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=len(present), frameon=False,
              fontsize=8.5, labelcolor=SECONDARY, handletextpad=0.3, columnspacing=1.1)
    degree = max(len(ps) for ps in parents)
    ax.set_title(f"{name}: {n} nodes, up to {degree} parent(s) per node, target = node {target}", color=INK, fontsize=11, pad=36)

    out_path = out_path or os.path.join(data_dir, f"{name}_graph.png")
    fig.savefig(out_path, dpi=150, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved graph to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="name of the stored table")
    parser.add_argument("--data_dir", help="default: $CUSTOM_SCM_DATA or the lustre folder")
    parser.add_argument("--out", help="path of the png, default next to the table")
    args = parser.parse_args()
    plot_graph(args.name, data_dir=args.data_dir, out_path=args.out)


if __name__ == "__main__":
    main()
