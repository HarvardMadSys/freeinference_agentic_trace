"""The tool overview: the native and shell split, example commands, the per-category table, and latency and result size."""

import pathlib

import pandas
from matplotlib import lines, patches

from paper.metrics import tools
from paper.plot_style import axes, style, tables
from release import schema

OVERVIEW_COLUMNS = ("calls", "sessions", "tokens", "time", "fail", "same", "mixed")
OVERVIEW_HEADER = ["Category", "Calls", "Sess.", "Tokens", "Time", "Fail", "Same", "Mixed"]
OVERVIEW_POSITIONS = tuple((0.02 if index == 0 else 0.33 + 0.095 * index, "left" if index == 0 else "right")
                           for index in range(len(OVERVIEW_HEADER)))
EXAMPLE_POSITIONS = ((0.02, "left"), (0.27, "left"))
TABLE_WIDTH = 5.0  # inches: seven number columns need more than one column of the paper
EXAMPLES_TABLE_WIDTH = 5.2  # inches: eight programs per row need more than a column
EXAMPLE_CHARACTERS_PER_LINE = 54  # a row's examples wrap onto a new line before passing this many characters, as the paper wraps them
MIN_VALUES_DRAWN = 30  # a category with fewer values is not drawn
MIN_LATENCY_DRAWN_MS = 100  # the latency axis starts at 100 ms, as in the paper
MIN_TOKENS_DRAWN = 1  # the tokens axis starts at 1 token, below the shortest success messages (4 and 5 tokens); empty results are drawn at 1
WINDOW_HEADROOM = 1.6  # the axis runs this far past the largest whisker or mean


# ---- the marks of a box plot's row: only these two figures draw one
BOX_HEIGHT = 0.6  # a box row fills this much of its row's height


MEAN_MARK = {"marker": "D", "ms": 6.5, "mfc": "white", "mec": "0.1", "mew": 1.1, "ls": "none"}  # the mean's diamond


def box_row(axes, position: float, summary: dict, color: str):
    """One row of a box plot: whiskers with caps from P5 to P95, a box from P25 to P75 filled with the row's colour
    tinted and edged with it shaded, a black median, and a diamond at the mean."""
    edge = style.shade(color)
    axes.plot([summary["p5"], summary["p95"]], [position, position], color=edge, lw=1.0, zorder=2)
    for end in ("p5", "p95"):
        axes.plot([summary[end]] * 2, [position - BOX_HEIGHT / 4, position + BOX_HEIGHT / 4], color=edge, lw=1.0)
    box = patches.Rectangle((summary["p25"], position - BOX_HEIGHT / 2), summary["p75"] - summary["p25"], BOX_HEIGHT,
                            facecolor=style.tint(color), edgecolor=edge, lw=1.0, zorder=3)
    axes.add_patch(box)
    axes.plot([summary["p50"]] * 2, [position - BOX_HEIGHT / 2, position + BOX_HEIGHT / 2], color="black", lw=1.2, zorder=4)
    axes.plot([summary["mean"]], [position], zorder=5, **MEAN_MARK)


def mean_key(axes):
    """The key of the mean's diamond, at the lower right inside the plot."""
    handle = lines.Line2D([], [], **MEAN_MARK)
    axes.legend([handle], ["Mean"], loc="lower right", handlelength=1.0, handletextpad=0.3, borderaxespad=0.2)


def row_color(category: str) -> str:
    """A shell category's rows are drawn gold, the others grey."""
    return style.SHELL_TOOL_COLOR if category.startswith(tools.SHELL_CATEGORY_MARK) else style.NATIVE_TOOL_COLOR


def shell_split(calls: pandas.DataFrame) -> dict:
    """The share of calls that are shell calls."""
    return {"shell_share": tools.shell_share(calls)}


def wrapped_examples(labels: list[str]) -> list[str]:
    """The examples joined by commas, in lines of at most EXAMPLE_CHARACTERS_PER_LINE characters, never splitting a
    command; one empty line when there are none."""
    lines = [""]
    for label in labels:
        piece = label if not lines[-1] else f"{lines[-1]}, {label}"
        if lines[-1] and len(piece) > EXAMPLE_CHARACTERS_PER_LINE:
            lines[-1] += ","
            lines.append(label)
        else:
            lines[-1] = piece
    return lines


def example_commands(calls: pandas.DataFrame, output_directory: pathlib.Path) -> dict:
    """The example commands: the most frequent programs of each shell category's single-program calls."""
    examples = tools.example_commands(calls)
    lines = [("rule", []), ("header", ["Category", "Example commands"]), ("rule", [])]
    for category, labels in examples.items():
        wrapped = wrapped_examples(labels)
        lines.append(("row", [category, wrapped[0]]))
        lines.extend(("row", ["", line]) for line in wrapped[1:])
    lines.append(("rule", []))
    columns = {"header": EXAMPLE_POSITIONS, "row": EXAMPLE_POSITIONS}
    tables.draw_table(lines, columns, EXAMPLES_TABLE_WIDTH, frozenset({"header"}), output_directory, "example_commands")
    pandas.DataFrame([{"category": category, "examples": ", ".join(labels)} for category, labels in examples.items()]).to_csv(
        output_directory / "example_commands.csv", index=False)
    return {f"examples_{category}": labels for category, labels in examples.items()}


def overview_table(calls: pandas.DataFrame, session_count: int, output_directory: pathlib.Path) -> dict:
    """The tool overview: per category, its shares of calls, sessions, result tokens and wait, failure rate and batching."""
    table = tools.overview_table(calls, session_count)
    lines = [("rule", []), ("header", OVERVIEW_HEADER), ("rule", [])]
    for category, row in table.iterrows():
        kind = "all" if category == "All" else "row"
        if kind == "all":
            lines.append(("rule", []))
        lines.append((kind, [category] + [f"{row[column]:.1f}" for column in OVERVIEW_COLUMNS]))
    lines.append(("rule", []))
    columns = {"header": OVERVIEW_POSITIONS, "row": OVERVIEW_POSITIONS, "all": OVERVIEW_POSITIONS}
    tables.draw_table(lines, columns, TABLE_WIDTH, frozenset({"header", "all"}), output_directory, "tool_overview")
    table.to_csv(output_directory / "tool_overview.csv", index_label="category")
    numbers = {}
    for category, row in table.iterrows():
        for column in OVERVIEW_COLUMNS:
            numbers[f"{category}_{column}_percent"] = float(row[column])
    return numbers


def box_rows(summaries: dict[str, dict], order: list[str], output_directory: pathlib.Path, name: str, quantity: str,
             noun: str, low: float) -> list[str]:
    """One box row per category with enough values, top to bottom in `order`, and the mean's key; return the
    categories not drawn. The axis runs from `low` to past the largest whisker or mean."""
    drawn = [category for category in order if summaries[category]["n"] >= MIN_VALUES_DRAWN]
    high = max(max(summaries[category]["p95"], summaries[category]["mean"]) for category in drawn) * WINDOW_HEADROOM
    figure, plot = style.new_figure(tall=True)
    positions = list(range(len(drawn)))[::-1]
    for category, position in zip(drawn, positions):
        clipped = {key: max(value, low) for key, value in summaries[category].items() if key != "n"}
        box_row(plot, position, clipped, row_color(category))
    axes.category_axis(plot, "y", positions, drawn)
    plot.set_ylim(-0.7, len(drawn) - 0.3)
    axes.apply(plot, "x", quantity, noun, low=low, high=high)
    plot.grid(False, axis="y", which="both")
    mean_key(plot)
    style.save(figure, output_directory, name)
    return [category for category in order if category not in drawn]


def in_milliseconds(summary: dict) -> dict:
    """A latency summary with its values in milliseconds instead of seconds."""
    return {key: value if key == "n" else value * schema.MILLISECONDS_PER_SECOND for key, value in summary.items()}


def latency_and_result_size(calls: pandas.DataFrame, output_directory: pathlib.Path) -> dict[str, dict]:
    """Two figures, each category's tool wait and its result tokens, and the numbers of each by its file name."""
    latency = tools.latency_by_category(calls)
    tokens = tools.result_tokens_by_category(calls)
    latency_ms = {category: in_milliseconds(summary) for category, summary in latency.items()}
    order = tools.category_order(calls)
    undrawn_latency = box_rows(latency_ms, order, output_directory, "latency_by_category", "latency_ms", "latency",
                               MIN_LATENCY_DRAWN_MS)
    undrawn_tokens = box_rows(tokens, order, output_directory, "result_tokens_by_category", "tokens", "tokens",
                              MIN_TOKENS_DRAWN)
    latency_numbers = {"not_drawn": undrawn_latency}
    tokens_numbers = {"not_drawn": undrawn_tokens}
    for category in schema.TOOL_CATEGORIES:
        latency_numbers[f"{category}_median_latency_s"] = latency[category]["p50"]
        latency_numbers[f"{category}_mean_latency_s"] = latency[category]["mean"]
        latency_numbers[f"{category}_p95_latency_s"] = latency[category]["p95"]
        tokens_numbers[f"{category}_median_result_tokens"] = tokens[category]["p50"]
    return {"latency_by_category": latency_numbers, "result_tokens_by_category": tokens_numbers}
