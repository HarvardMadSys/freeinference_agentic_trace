"""Context mutations: how often each happens and what prefill it causes (a table), and the hit ratio after one."""

import pathlib

import pandas

from paper.plot_style import axes, marks, style, tables
from paper.metrics import mutations

TABLE_WIDTH = 4.2  # inches
TABLE_COLUMNS = ((0.03, "left"), (0.58, "right"), (0.78, "right"), (0.97, "right"))
CAUSE_INDENT = "    "  # a cause is set under its transition, indented


def table_lines(table: pandas.DataFrame) -> list[tuple[str, list[str]]]:
    """The table's lines: a header, the two transitions in bold, then the causes of a mutation, indented."""
    lines = [("rule", []), ("header", ["Transition", "Steps (%)", "Sess. (%)", "Prefill (%)"]), ("rule", [])]
    for row in table.itertuples():
        kind = "transition" if row.row in ("append", "mutation") else "cause"
        label = row.label if kind == "transition" else CAUSE_INDENT + row.label
        cells = [label, f"{100 * row.steps_share:.2f}", f"{100 * row.sessions_share:.1f}", f"{100 * row.prefill_share:.1f}"]
        lines.append((kind, cells))
    lines.append(("rule", []))
    return lines


def mutation_table(frame: pandas.DataFrame, output_directory: pathlib.Path) -> dict:
    """The mutation table: context mutations by cause, their shares of steps, sessions and prefill with no TTL."""
    table = mutations.mutation_table(frame)
    columns = {"header": TABLE_COLUMNS, "transition": TABLE_COLUMNS, "cause": TABLE_COLUMNS}
    tables.draw_table(table_lines(table), columns, TABLE_WIDTH, frozenset({"header", "transition"}),
                      output_directory, "mutations_by_cause")
    table.to_csv(output_directory / "mutations_by_cause.csv", index=False)
    numbers = {"transitions": int(len(frame))}
    for row in table.itertuples():
        numbers[f"{row.row}_steps_percent"] = 100 * row.steps_share
        numbers[f"{row.row}_sessions_percent"] = 100 * row.sessions_share
        numbers[f"{row.row}_prefill_percent"] = 100 * row.prefill_share
    return numbers


def hit_ratio_after_transition(frame: pandas.DataFrame, output_directory: pathlib.Path) -> dict:
    """CDFs of the provider-reported hit ratio of requests after an append, and after a mutation."""
    figure, plot = style.new_figure()
    numbers = {}
    for transition, label in (("append", "Append"), ("mutation", "Mutation")):
        values = frame.hit_ratio[(frame.transition == transition) & frame.hit_ratio.notna()]
        marks.cdf_line(plot, values, label, style.TRANSITION_COLOR[label])
        numbers[f"median_hit_ratio_{transition}"] = float(values.median())
    axes.apply(plot, "x", "ratio", "hit ratio")
    axes.apply(plot, "y", "cdf", "requests")
    marks.legend_inside(plot, "upper left")
    style.save(figure, output_directory, "hit_ratio_after_transition")
    return numbers
