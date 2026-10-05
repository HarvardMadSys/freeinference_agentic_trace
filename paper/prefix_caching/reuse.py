"""Prefix caching: how the cache hit ratio falls as the gap before a request grows, and what the gap waited for."""

import pathlib

from release import reader
from paper.plot_style import axes, marks, style
from paper.metrics import binning, gaps

MIN_GAP_DRAWN_S = 0.1  # a gap shorter than this is drawn at it, on the log axis
MAX_GAP_DRAWN_S = 86_400
MEAN_HIT_RATIO_AXIS_LOW = 0.7  # the bars start here, as in the paper, so their differences show
GAP_BIN_LEGEND_COLUMNS = 3
MIN_IDLE_GAPS_PER_PROVIDER = 3_000  # a provider with fewer measured idle gaps is not drawn
PROVIDERS_DRAWN = 4  # the providers with the most measured idle gaps
PROVIDER_COLORS = (style.BLUE, style.GREEN, style.RED, style.ORANGE)


def hit_ratio_by_gap_bin(population: reader.Release, output_directory: pathlib.Path) -> dict:
    """CDFs of the hit ratio of requests in each gap bin; a bin with no rows is not drawn."""
    rows = gaps.gap_rows(population.requests).dropna(subset=["hit_ratio"])
    figure, plot = style.new_figure(top_legend=True)
    numbers = {}
    for name, label, _, _ in gaps.GAP_BINS:
        values = rows.hit_ratio[rows.gap_bin == label]
        median = binning.median(values) if len(values) else None
        numbers[f"median_hit_ratio_{name}"] = median
        if len(values):
            marks.cdf_line(plot, values, label, style.GAP_BIN_COLOR[label])
    axes.apply(plot, "x", "ratio", "hit ratio")
    axes.apply(plot, "y", "cdf", "requests")
    marks.legend_above(plot, columns=GAP_BIN_LEGEND_COLUMNS)
    style.save(figure, output_directory, "hit_ratio_by_gap_bin")
    return numbers


def gap_by_cause(population: reader.Release, output_directory: pathlib.Path) -> dict:
    """CDFs of the gap before a request, by what it waited for; the tool and user-idle medians are marked."""
    rows = gaps.gap_rows(population.requests)
    figure, plot = style.new_figure()
    numbers = {}
    for waited_for, cause in gaps.causes_present(rows):
        values = rows.gap_s[rows.cause == cause]
        color = style.CAUSE_COLOR[cause]
        marks.cdf_line(plot, values, cause, color, floor=MIN_GAP_DRAWN_S)
        median = binning.median(values)
        numbers[f"median_gap_s_{waited_for}"] = median
        if waited_for == "tool":
            marks.median_label(plot, median, color, marks.duration_text(median), (-4, 30), "right")
        if waited_for == "user":
            marks.median_label(plot, median, color, marks.duration_text(median), (12, 27), "left")
    axes.apply(plot, "x", "duration", "time", low=MIN_GAP_DRAWN_S, high=MAX_GAP_DRAWN_S)
    axes.apply(plot, "y", "cdf", "requests")
    marks.legend_inside(plot, "lower right")
    style.save(figure, output_directory, "gap_by_cause")
    return numbers


def mean_hit_ratio_by_cause(population: reader.Release, output_directory: pathlib.Path) -> dict:
    """The mean hit ratio of requests by what the gap before them waited for."""
    rows = gaps.gap_rows(population.requests)
    means_by_cause = gaps.mean_hit_ratio_by_cause(rows)
    pairs = gaps.causes_present(rows.dropna(subset=["hit_ratio"]))
    causes = list(means_by_cause.index)
    means = list(means_by_cause)
    positions = list(range(len(causes)))[::-1]
    figure, plot = style.new_figure()
    marks.horizontal_bars(plot, positions, means, [style.CAUSE_COLOR[cause] for cause in causes],
                          left=MEAN_HIT_RATIO_AXIS_LOW, value_labels=[f"{mean:.2f}" for mean in means])
    axes.category_axis(plot, "y", positions, causes)
    plot.set_ylim(min(positions) - 0.55, max(positions) + 0.55)
    axes.apply(plot, "x", "ratio_top", "hit ratio", low=MEAN_HIT_RATIO_AXIS_LOW, high=1.0)
    plot.grid(False, which="both")  # bars read against their own labels, as in the paper
    style.save(figure, output_directory, "mean_hit_ratio_by_cause")
    return {f"mean_hit_ratio_{waited_for}": mean
            for (waited_for, _), mean in zip(pairs, means)}


def retention_by_provider(population: reader.Release, output_directory: pathlib.Path) -> dict:
    """The median hit ratio against the idle gap at the provider, for the providers with the most idle gaps.

    Each point is a group of idle-gap bins holding enough rows, drawn at the
    median gap of its rows.
    """
    rows = gaps.idle_gap_rows(population.requests)
    counts = rows.provider.value_counts()
    largest = counts[counts >= MIN_IDLE_GAPS_PER_PROVIDER].index[:PROVIDERS_DRAWN]
    figure, plot = style.new_figure()
    numbers = {}
    for provider, color in zip(sorted(largest), PROVIDER_COLORS):
        points = gaps.retention_points(rows[rows.provider == provider])
        plot.plot(points.gap_s, points.hit_ratio, color=color, lw=style.LINE_WIDTH,
                  label=f"Prov {provider}")
        numbers[f"median_hit_ratio_longest_gap_provider_{provider}"] = float(points.hit_ratio.iloc[-1])
    axes.apply(plot, "x", "gap", "time", low=gaps.IDLE_GAP_EDGES_S[0], high=gaps.IDLE_GAP_EDGES_S[-1])
    axes.apply(plot, "y", "ratio", "hit ratio")
    if len(largest):
        marks.legend_inside(plot, "lower left")
    style.save(figure, output_directory, "retention_by_provider")
    return numbers
