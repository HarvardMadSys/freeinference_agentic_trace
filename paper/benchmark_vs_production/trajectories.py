"""Tool use along a turn, for the benchmark and for human-driven sessions: the share of calls in each group per tenth."""

import pathlib

import pandas

from release import reader
from paper.plot_style import axes, marks, style
from paper.metrics import trajectories

PERCENT_PER_BIN = 100 // trajectories.BINS


def trajectory_mix(tool_calls: pandas.DataFrame, release: reader.Release, output_directory: pathlib.Path, name: str) -> dict:
    """The share of each group of calls at each tenth of the turn, stacked."""
    shares = trajectories.trajectory_shares(tool_calls, release.requests)
    x = [(tenth + 1) * PERCENT_PER_BIN for tenth in shares.index]
    bands = [(group, shares[group].to_numpy(), style.TRAJECTORY_COLOR[group]) for group in trajectories.GROUPS]
    figure, plot = style.new_figure(top_legend=True)
    marks.stack_area(plot, x, bands)
    axes.apply(plot, "x", "trajectory", "turn trajectory", low=PERCENT_PER_BIN, high=100)
    axes.apply(plot, "y", "share", "calls", low=0, high=1)
    plot.grid(False, axis="x", which="both")
    marks.legend_above(plot, columns=4, font_size=style.FONT_LEGEND_DENSE, compact=True)
    style.save(figure, output_directory, name)
    first, last = shares.iloc[0], shares.iloc[-1]
    return {f"{group}_share_first_tenth": float(first[group]) for group in trajectories.GROUPS} | {
        f"{group}_share_last_tenth": float(last[group]) for group in trajectories.GROUPS}
