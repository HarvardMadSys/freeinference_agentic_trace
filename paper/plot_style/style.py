"""How every figure looks: the canvas, the font sizes, the palette by meaning, and saving.

These follow the paper's own plotting style, so a figure drawn here matches
the one printed in the paper. A figure module takes its canvas, font sizes
and palette from here, and types only the line widths and alphas of the marks
that are its own.
"""

import pathlib

from matplotlib import colors, pyplot

CANVAS = (3.5, 3.0)  # inches: every panel of the paper
CANVAS_TOP_LEGEND = (3.5, 3.55)  # the same plot region with room for a legend above it
CANVAS_TALL = (4.68, 5.78)  # a panel with one row per tool category, printed beside the tool table at its height
DPI = 300

# Font sizes in points on the canvas; the paper prints panels at about half size.
FONT_AXIS = 16.5
FONT_TICK = 14.0
FONT_CATEGORY = 13.5
FONT_LEGEND = 12.3
FONT_NOTE = 10.3
FONT_LEGEND_DENSE = 9.2  # a legend of eight entries in two rows: the legend size four 7 % steps down
FONT_CALLOUT = 13.5
LINE_WIDTH = 1.8

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": FONT_TICK,
    "axes.titlesize": FONT_AXIS,
    "axes.labelsize": FONT_AXIS,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "legend.fontsize": FONT_LEGEND,
    "lines.linewidth": LINE_WIDTH,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "grid.color": "0.8",
    "grid.alpha": 0.4,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "savefig.dpi": DPI,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

MINOR_GRID_ALPHA = 0.2  # minor grid lines are half as strong as the major ones

TIMES = "×"
INK = "0.15"  # near-black, for callouts
REFERENCE_GREY = "0.55"

# The palette: one hue per meaning across the paper.
BLUE = "#1f6feb"  # the model working: input, prefill, assistant
BLUE_LIGHT = "#6aa6f0"  # its lighter partner
GREEN = "#2da44e"  # output, tools
ORANGE = "#e8710a"  # decode
RED = "#d1242f"  # a person: user steps, human time; uncached input
BROWN = "#b5885a"  # the system prompt
PURPLE = "#8957e5"  # a person answering the agent's question
YELLOW = "#e3b21b"  # fresh input and storage: what a longer retention trades
GREY = "#5a6169"  # the grey dashed reference series

TURN_COLOR = {"LLM Requests": BLUE, "User Steps": RED, "Tool Steps": GREEN}
SPAN_COLOR = {"User Turn": RED, "Session": BLUE}
TOKEN_MIX_COLOR = {"Input": BLUE, "Input (Uncached)": RED, "Output": GREEN}
GROWTH_COLOR = {"Input (Cached)": BLUE, "Input (Uncached)": RED, "Output": GREEN}
TOKEN_SHARE_COLOR = {"Input (Cached)": BLUE, "Input (Uncached)": RED, "Output": GREEN}
ROLE_COLOR = {"System": BROWN, "System + Tools": BROWN, "User": RED, "Assistant": BLUE, "Tool": GREEN}
COMPONENT_COLOR = {"Prefill": BLUE, "Decode": ORANGE, "Tool": GREEN, "User Idle": RED, "Queueing": GREY}
CAUSE_COLOR = {"Tool Gap": GREEN, "Ask-User Tool": PURPLE, "User Idle": RED, "Subagent": BLUE}
SCOPE_COLOR = {"Human-driven": BLUE, "Benchmark": RED}
TRANSITION_COLOR = {"Append": GREEN, "Mutation": RED}
RETAINED_COLOR = {"Reusable": BLUE, "Dead": RED}
API_COST_COLOR = {"Output": GREEN, "Cache Read": BLUE, "Input": YELLOW}
INFRASTRUCTURE_COST_COLOR = {"Recompute": BLUE, "Storage": YELLOW}
POLICY_COLOR = {"ttl_only": GREY, "reclaim_mutate": BLUE, "reclaim_dead": RED, "reclaim_both": GREEN}
POLICY_NAMES = {"ttl_only": "TTL Only", "reclaim_mutate": "Reclaim Mutate", "reclaim_dead": "Reclaim Dead",
                "reclaim_both": "Reclaim Both"}
# Gap bins from short to long, light to dark: matplotlib's inferno map from 0.85 down to 0.08.
NATIVE_TOOL_COLOR = "#8c8c8c"  # a native tool's row
SHELL_TOOL_COLOR = "#bf8700"  # a shell category's row
BOX_TINT = 0.55  # a box is filled with its row's colour mixed this far toward white
BOX_SHADE = 0.7  # and edged with its row's colour darkened to this share
TRAJECTORY_COLOR = {"Read": BLUE, "Search": "#16b5e0", "Write/Edit": GREEN, "Execute": RED, "Build/Test": "#e6a700",
                    "Git/VCS": "#8250df", "Browser/Web": "#fa4549", "Other": "#6e7781"}
BATCH_COLOR = {"Serial + LLM Turn": GREY, "Batched Tool Latency": "#1a7f37"}
GAP_BIN_COLOR = {"<1m": "#fbbe23", "1–5m": "#f68013", "5–15m": "#d74b3f",
                 "15–30m": "#a82e5f", "30m–1h": "#6a176e", ">1h": "#10092d"}


def tint(color: str, share: float = BOX_TINT) -> tuple:
    """A colour mixed `share` of the way toward white."""
    red, green, blue = colors.to_rgb(color)
    return (red + (1 - red) * share, green + (1 - green) * share, blue + (1 - blue) * share)


def shade(color: str, share: float = BOX_SHADE) -> tuple:
    """A colour darkened to `share` of its brightness."""
    red, green, blue = colors.to_rgb(color)
    return (red * share, green * share, blue * share)


def new_figure(top_legend: bool = False, tall: bool = False):
    """A figure on the paper's canvas, with a legend band above the plot, or a taller plot, when asked."""
    pyplot.switch_backend("Agg")  # files only, never a window
    pyplot.rcParams.update(RC)
    size = CANVAS_TALL if tall else CANVAS_TOP_LEGEND if top_legend else CANVAS
    figure, axes = pyplot.subplots(figsize=size, layout="constrained")
    if top_legend:
        axes.set_box_aspect(CANVAS[1] / CANVAS[0])
    return figure, axes


def new_page(size: tuple[float, float]):
    """A figure with no axes, for a table set as text."""
    pyplot.switch_backend("Agg")  # files only, never a window
    pyplot.rcParams.update(RC)
    return pyplot.figure(figsize=size)


# No creation date in a file, so the same figure drawn again is the same bytes.
NO_DATE = {".pdf": {"CreationDate": None}, ".png": {}}


def save(figure, output_directory: pathlib.Path, name: str) -> list[pathlib.Path]:
    """Write the figure as `<name>.pdf` and `<name>.png`, and close it."""
    output_directory.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".pdf", ".png"):
        path = output_directory / f"{name}{suffix}"
        figure.savefig(path, dpi=DPI, metadata=NO_DATE[suffix])
        written.append(path)
    pyplot.close(figure)
    return written
