"""What today's harness pays as a multiple of a harness that mutates less, as cache reads get cheaper.

Every quantity comes from `metrics.cache_costs.cheaper_reads_curves`, over the
cheaper-reads model's sums added across the weeks. A multiple above 1 means the harness
that mutates less pays less.
"""

import math
import pathlib

import pandas

from paper.metrics import cache_costs
from paper.plot_style import axes, marks, style

# Each TTL's line: its label and colour.
TTL_LINES = (("5m", "5-min", style.GREEN), ("15m", "15-min", style.BLUE), ("none", "Infinite", style.ORANGE))
WINDOW_STEP = 0.05  # the y window opens one step of this size beyond the curves, on a twentieth of 1×


def window(values: pandas.Series) -> tuple[float, float]:
    """The y window: from below the lowest value, and 1×, to above the highest, on twentieths of 1×."""
    low = (math.floor(min(values.min(), 1.0) / WINDOW_STEP) - 1) * WINDOW_STEP
    high = (math.ceil(max(values.max(), 1.0) / WINDOW_STEP) + 1) * WINDOW_STEP
    return low, high


def panel(curves: pandas.DataFrame, column: str, noun: str, output_directory: pathlib.Path, name: str) -> None:
    """One multiple against the read price reduction, one line per TTL, with 1× marked."""
    figure, plot = style.new_figure()
    for ttl, label, color in TTL_LINES:
        line = curves[curves.ttl == ttl].sort_values("read_price_cut")
        plot.plot(line.read_price_cut, line[column], color=color, lw=style.LINE_WIDTH, label=label)
    marks.reference_line(plot, 1.0)
    low, high = window(curves[column])
    axes.apply(plot, "x", "ratio", "read price reduction")
    axes.apply(plot, "y", "multiple_near_one", noun, low=low, high=high)
    marks.legend_inside(plot, "upper left").set_title("Retention Time")
    style.save(figure, output_directory, name)


def prefill_numbers(line: pandas.DataFrame, ttl: str) -> dict:
    """One TTL's numbers of the prefill panel, keyed by the TTL."""
    free_reads = line.sort_values("read_price_cut").iloc[-1]
    return {f"prefill_multiple_at_free_reads_{ttl}": float(free_reads.prefill_multiple),
            f"lowest_prefill_multiple_{ttl}": float(line.prefill_multiple.min())}


def cost_numbers(line: pandas.DataFrame, ttl: str) -> dict:
    """One TTL's numbers of the API cost panel, keyed by the TTL."""
    free_reads = line.sort_values("read_price_cut").iloc[-1]
    return {f"cost_multiple_at_free_reads_{ttl}": float(free_reads.cost_multiple),
            f"break_even_read_price_{ttl}": cache_costs.break_even_read_price(line)}


def draw(curves: pandas.DataFrame, output_directory: pathlib.Path) -> dict[str, dict]:
    """Both panels, prefill and API cost, and each panel's numbers per TTL under its file name."""
    panel(curves, "prefill_multiple", "prefill reduction", output_directory, "cheaper_reads_prefill")
    panel(curves, "cost_multiple", "API cost reduction", output_directory, "cheaper_reads_cost")
    numbers = {"cheaper_reads_prefill": {}, "cheaper_reads_cost": {}}
    for ttl, _, _ in TTL_LINES:
        line = curves[curves.ttl == ttl]
        numbers["cheaper_reads_prefill"].update(prefill_numbers(line, ttl))
        numbers["cheaper_reads_cost"].update(cost_numbers(line, ttl))
    return numbers
