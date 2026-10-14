"""The colours and markers shared by every figure of the project, and the look of an axis."""

INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
BACKGROUND = "#fcfcfb"

# The layers are ordered by depth, so they wear one hue that gets darker with depth. It
# starts at step 250 (2.06:1 on the surface), the lightest an ordered mark may be. Up to 5
# layers the steps picked from this ramp clear the validator's ordinal checks (monotone
# lightness, adjacent gaps >= 0.06 in OKLCH L); a single hue cannot go further than that and
# stay inside the ordinal contrast floor, so past 5 layers, color alone cannot carry the
# distinction any more - MARKERS is the redundant channel that keeps working past that
# ceiling (shape, not lightness), paired with color one-to-one by depth index.
BLUE_RAMP = (
    "#86b6ef",  # 250
    "#6da7ec",  # 300
    "#5598e7",  # 350
    "#3987e5",  # 400
    "#2a78d6",  # 450
    "#256abf",  # 500
    "#1c5cab",  # 550
    "#184f95",  # 600
    "#104281",  # 650
    "#0d366b",  # 700
)
# The same ramp in the next two categorical hues (orange, aqua), one step per step of the
# blue one and at the same OKLab lightness (so the ordinal contrast is the one of BLUE_RAMP),
# for the models whose layers belong to modules.
ORANGE_RAMP = (
    "#eb9c7f", "#e68764", "#df7249", "#d95923", "#c94908",
    "#b44005", "#9e3601", "#8a2e00", "#742703", "#611e01",
)
AQUA_RAMP = (
    "#80c3a2", "#62b890", "#43ac7f", "#049f6d", "#068f62",
    "#037f56", "#096f4b", "#066041", "#035136", "#00432b",
)
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "h", "8")

# The role of a node of a causal graph towards the target: colour and marker, so the roles do
# not rest on hue alone.
ROLE_STYLE = {
    "target": (INK, "D"),
    "parent": ("#2a78d6", "o"),
    "child": ("#eb6834", "s"),
    "spouse": ("#1baf7a", "h"),
    "outside": (MUTED, "^"),
}


def style_axes(ax, grid_axis="both"):
    """The surface, grid, ticks and spines of every panel."""
    ax.set_facecolor(BACKGROUND)
    ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_COLOR)
