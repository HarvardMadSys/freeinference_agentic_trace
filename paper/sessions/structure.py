"""Sessions: how many steps a session has, and how long a turn and a session last."""

import pathlib

from release import reader
from paper.plot_style import axes, marks, style
from paper.metrics import binning, turns

MIN_TURN_S = 0.1  # a turn shorter than this is drawn at it, on the log axis
MIN_SESSION_S = 1.0  # a session shorter than this is drawn at it


def counts_per_session(population: reader.Release, output_directory: pathlib.Path) -> dict:
    """CDFs of requests, user steps and tool steps per session."""
    counts = turns.counts_per_session(population.requests)
    figure, plot = style.new_figure(top_legend=True)
    series = [("requests", "LLM Requests", (18, -30), "left"),
              ("user_steps", "User Steps", (-10, 28), "right"),
              ("tool_steps", "Tool Steps", (-18, 30), "right")]
    numbers = {}
    for column, label, offset, align in series:
        color = style.TURN_COLOR[label]
        marks.cdf_line(plot, counts[column], label, color)
        median = binning.median(counts[column])
        marks.median_label(plot, median, color, marks.count_text(median), offset, align)
        numbers[f"median_{column}"] = median
    axes.apply(plot, "x", "count", "count per session", low=1, high=1000)
    axes.apply(plot, "y", "cdf", "sessions")
    marks.legend_above(plot)
    style.save(figure, output_directory, "counts_per_session")
    return numbers


def turn_and_session_duration(population: reader.Release, output_directory: pathlib.Path) -> dict:
    """CDFs of how long one user turn lasts and how long a whole session lasts."""
    turn_s = turns.turn_durations(population.requests)
    session_s = turns.session_durations(population.requests)
    figure, plot = style.new_figure(top_legend=True)
    marks.cdf_line(plot, turn_s, "User Turn", style.SPAN_COLOR["User Turn"], floor=MIN_TURN_S)
    marks.cdf_line(plot, session_s, "Session", style.SPAN_COLOR["Session"], floor=MIN_SESSION_S)
    turn_median = binning.median(turn_s)
    session_median = binning.median(session_s)
    marks.median_label(plot, turn_median, style.SPAN_COLOR["User Turn"], marks.duration_text(turn_median))
    marks.median_label(plot, session_median, style.SPAN_COLOR["Session"], marks.duration_text(session_median),
                       (12, -27), "left")
    axes.apply(plot, "x", "duration", "duration", low=1, high=604_800)
    axes.apply(plot, "y", "cdf", "sessions")
    marks.legend_above(plot)
    style.save(figure, output_directory, "turn_and_session_duration")
    return {
        "turn_median_s": turn_median,
        "session_median_s": session_median,
        "session_p90_s": float(session_s.quantile(0.9)),
    }
