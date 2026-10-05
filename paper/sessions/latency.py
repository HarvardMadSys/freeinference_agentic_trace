"""Latency and time: TTFT against decode and queueing for two models, where session time goes, and the time saved by faster models."""

import pathlib

import numpy

from release import reader
from paper.plot_style import axes, marks, style
from paper.metrics import binning, latency

MODELS = ("minimax-m2.7", "kimi-k2.7-code")  # the two externally routed models the paper decomposes
MIN_LATENCY_DRAWN_S = 1e-3  # a latency below 1 ms is drawn at it
SPEEDUP_AXIS_HEADROOM = 1.26  # the speedup axis runs this far past the largest bar, leaving room for its label
TIME_PARTS = ("human", "tool", "decode", "prefill")  # the parts of a session's time, in the order the paper lists them


# ---- several legends inside one plot
def legend_groups(plot, groups: list[tuple[list[str], str]]):
    """Several legends inside the plot, each of the named series at its own location."""
    handles, labels = plot.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    for labels_of_group, location in groups:
        legend = plot.legend([by_label[label] for label in labels_of_group], labels_of_group, loc=location,
                             handlelength=1.2, handletextpad=0.4, labelspacing=0.3)
        plot.add_artist(legend)


def ttft_decomposition(population: reader.Release, output_directory: pathlib.Path, model: str) -> dict:
    """CDFs of one model's TTFT, decode time and queueing, over requests with a trusted TTFT."""
    parts = latency.ttft_decomposition(population.requests, model)
    if parts is None:
        return {"drawn": f"no: fewer than {latency.MIN_LONG_MISSES} long misses"}
    figure, plot = style.new_figure()
    series = [("ttft", "TTFT", style.COMPONENT_COLOR["Prefill"], False, (12, 27), "left"),
              ("decode", "Decode", style.COMPONENT_COLOR["Decode"], False, (12, -27), "left"),
              ("queueing", "Queueing", style.COMPONENT_COLOR["Queueing"], True, (-12, 27), "right")]
    numbers = {"prefill_rate_tokens_per_s": parts["rate"],
               "long_misses": parts["long_misses"]}
    for name, label, color, dashed, offset, align in series:
        marks.cdf_line(plot, parts[name], label, color, floor=MIN_LATENCY_DRAWN_S, dashed=dashed)
        median = binning.median(parts[name])
        marks.median_label(plot, median, color, marks.duration_text(median), offset, align)
        numbers[f"median_{name}_s"] = median
    axes.apply(plot, "x", "latency", "latency", low=0.1, high=100)
    axes.apply(plot, "y", "cdf", "requests")
    legend_groups(plot, [(["TTFT", "Decode"], "upper left"), (["Queueing"], "lower right")])
    style.save(figure, output_directory, f"ttft_decomposition_{model}")
    return numbers


def session_time_breakdown(population: reader.Release, output_directory: pathlib.Path) -> dict:
    """Each session's time split between prefill, decode, tools and a person, in rank bins."""
    shares = latency.time_shares(population.requests)
    binned = binning.rank_bins(shares, ["human", "tool"])
    bands = [("Prefill", binned.prefill, style.COMPONENT_COLOR["Prefill"]),
             ("Decode", binned.decode, style.COMPONENT_COLOR["Decode"]),
             ("Tool", binned.tool, style.COMPONENT_COLOR["Tool"]),
             ("User Idle", binned.human, style.COMPONENT_COLOR["User Idle"])]
    figure, plot = style.new_figure(top_legend=True)
    marks.stack_area(plot, binned.percentile, bands)
    axes.apply(plot, "x", "percentile", low=0, high=100)
    axes.apply(plot, "y", "share", "time", low=0, high=1)
    plot.grid(False, axis="x", which="both")
    marks.legend_above(plot)
    style.save(figure, output_directory, "session_time_breakdown")
    means = shares.mean()
    return {f"mean_{part}_share": float(means[part]) for part in TIME_PARTS}


def speedup_if_twice_as_fast(population: reader.Release, output_directory: pathlib.Path) -> dict:
    """The session-time speedup if prefill, decode or tools alone ran twice as fast, tool waits capped."""
    speedups = latency.speedups(latency.time_breakdown(population.requests, cap_tool_waits=True))
    parts = [("prefill", "Prefill", "Prefill"), ("decode", "Decode", "Decode"), ("tool", "Tools", "Tool")]
    values = [speedups[name] for name, _, _ in parts]
    positions = list(numpy.arange(len(parts))[::-1])
    figure, plot = style.new_figure()
    marks.horizontal_bars(plot, positions, values, [style.COMPONENT_COLOR[color] for _, _, color in parts],
                          left=1.0, value_labels=[f"{value:.2f}{style.TIMES}" for value in values])
    axes.category_axis(plot, "y", positions, [f"{label} 2{style.TIMES}" for _, label, _ in parts])
    plot.set_ylim(min(positions) - 0.55, max(positions) + 0.55)
    axes.apply(plot, "x", "speedup", "time speedup", low=1.0, high=1.0 + (max(values) - 1.0) * SPEEDUP_AXIS_HEADROOM)
    plot.grid(False, which="both")  # bars read against their own labels, as in the paper
    style.save(figure, output_directory, "speedup_if_twice_as_fast")
    return {f"{name}_speedup": speedups[name] for name, _, _ in parts}
