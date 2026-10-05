"""The trace summary, set as the paper sets it: the totals, then sessions and input tokens per harness.

The quantities are `metrics.summary`'s. The table is written as JSON and CSV,
and as PDF and PNG.
"""

import json
import pathlib

import pandas
from release import population, reader
from paper.metrics import summary
from paper.plot_style import tables


HARNESS_NAMES = {
    "hermes": "Hermes", "pi": "Pi", "oh-my-pi": "Oh-My-Pi", "claude-code": "Claude Code", "opencode": "OpenCode",
    population.BENCHMARK_HARNESS: "SWE-Bench", "kilo": "Kilo", "codex": "Codex", "copilot": "Copilot",
    "zcode": "Zcode", "openclaw": "OpenClaw", "cline": "Cline", "zoo": "Zoo", "openhands": "OpenHands",
}

# The table is set in the paper's table font size on a page wide enough for its
# cells; like the paper's, it is scaled down to the column width when included.
TABLE_WIDTH = 5.0  # inches
# Where each cell sits, as a share of the width, and which side it aligns to.
SUMMARY_COLUMNS = ((0.02, "left"), (0.47, "right"), (0.53, "left"), (0.98, "right"))
HARNESS_COLUMNS = ((0.02, "left"), (0.27, "right"), (0.47, "right"), (0.53, "left"), (0.79, "right"), (0.98, "right"))


def summary_cells(totals: dict, has_benchmark: bool) -> list[list[str]]:
    """The totals as (label, value) cells, in the paper's order and number forms."""
    harnesses = f"{totals['harnesses']} (+1)" if has_benchmark else str(totals["harnesses"])
    return [
        ["Time span", f"{totals['months']} months"], ["Sessions", f"{totals['sessions']:,}"],
        ["Requests", f"{totals['requests']:,}"], ["Tool calls", f"{totals['tool_calls']:,}"],
        ["Input tokens", f"{totals['input_tokens'] / 1e9:.1f} B"], ["Output tokens", f"{totals['output_tokens'] / 1e6:.1f} M"],
        ["Cached tokens", f"{totals['cached_tokens'] / 1e9:.1f} B"], ["Agent harnesses", harnesses],
    ]


def harness_cells(harness_table: pandas.DataFrame) -> list[list[str]]:
    """Each harness as (name, sessions, input tokens) cells, the largest first."""
    cells = []
    for harness, row in harness_table.iterrows():
        cells.append([HARNESS_NAMES.get(harness, harness), f"{int(row.sessions):,}", f"{row.input_tokens / 1e9:.1f}B"])
    return cells


def in_pairs(cells: list[list[str]]) -> list[list[str]]:
    """Cells set two to a line, left then right, as the paper's two-column table reads."""
    pairs = []
    for index in range(0, len(cells), 2):
        right = cells[index + 1] if index + 1 < len(cells) else []
        pairs.append(cells[index] + right)
    return pairs


def table_rows(totals: dict, harness_table: pandas.DataFrame, has_benchmark: bool) -> list[tuple[str, list[str]]]:
    """The table's lines from top to bottom, each a kind (rule, title, summary, header, harness) and its cells."""
    rows = [("rule", []), ("title", ["Trace Summary"]), ("rule", [])]
    rows += [("summary", pair) for pair in in_pairs(summary_cells(totals, has_benchmark))]
    rows += [("rule", []), ("title", ["Sessions and Input Tokens by Harness"])]
    rows.append(("header", ["Harness", "Sess.", "Input Tokens", "Harness", "Sess.", "Input Tokens"]))
    rows += [("harness", pair) for pair in in_pairs(harness_cells(harness_table))]
    rows.append(("rule", []))
    return rows


def write_trace_summary(release: reader.Release, output_directory: pathlib.Path) -> dict:
    """Write the summary as JSON, the per-harness table as CSV, and the set table; return the totals beside the paper's."""
    everything = population.sessions_in_scope(release, population.ALL_SCOPES)
    totals = summary.totals(everything)
    harness_table = summary.by_harness(everything)
    output_directory.mkdir(parents=True, exist_ok=True)
    with open(output_directory / "trace_summary.json", "w") as handle:
        json.dump(totals, handle, indent=2)
        handle.write("\n")
    harness_table.to_csv(output_directory / "trace_summary_by_harness.csv")
    has_benchmark = population.BENCHMARK_HARNESS in harness_table.index
    columns = {"summary": SUMMARY_COLUMNS, "header": HARNESS_COLUMNS, "harness": HARNESS_COLUMNS}
    tables.draw_table(table_rows(totals, harness_table, has_benchmark), columns, TABLE_WIDTH, frozenset({"header"}),
                      output_directory, "trace_summary")
    return dict(totals)
