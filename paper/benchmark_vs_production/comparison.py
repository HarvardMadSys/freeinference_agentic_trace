"""Benchmark against production: human-driven (Agentic) and SWE-Bench sessions side by side.

Each figure is a CDF per scope with each median marked, a dotted line between
the medians, and their ratio, larger over smaller.
"""

import dataclasses
import pathlib
from typing import Callable

import pandas

from release import reader
from paper.plot_style import axes, marks, style
from paper.metrics import binning, tokens, turns

MIN_DURATION_DRAWN_S = 1.0  # a turn or session shorter than this is drawn at it
MIN_INPUT_DRAWN = 1_000  # the input axis starts at 1k, as in the paper
MIN_PROMPT_DRAWN = 1  # the user prompt axis starts at one token
SCOPE_LABELS = {"human": "Human-driven", "benchmark": "Benchmark"}


@dataclasses.dataclass(frozen=True)
class ComparisonAxis:
    """What one comparison is drawn against: the x quantity, its noun and window, how a median reads, the y noun."""

    quantity: str
    noun: str
    low: float
    high: float
    median_text: Callable[[float], str]
    y_noun: str


# ---- the ratio of two medians on one CDF
def median_ratio(axes, first: float, second: float) -> float:
    """A dotted line between two medians on a CDF, and their ratio, larger over smaller, in the lower right."""
    low, high = sorted((first, second))
    axes.plot([low, high], [0.5, 0.5], color=style.REFERENCE_GREY, lw=0.9, ls=":", zorder=7)
    axes.text(0.93, 0.1, f"{high / low:.1f}{style.TIMES}", transform=axes.transAxes, ha="right", va="bottom",
              fontsize=style.FONT_CALLOUT, color=style.INK)
    return high / low


def draw_comparison(human: pandas.Series, benchmark: pandas.Series, axis: ComparisonAxis, name: str,
                    output_directory: pathlib.Path) -> dict[str, float]:
    """The two CDFs, each median (the smaller labelled above left), and the ratio of the medians."""
    figure, plot = style.new_figure()
    medians = {}
    for key, values in (("human", human), ("benchmark", benchmark)):
        marks.cdf_line(plot, values, SCOPE_LABELS[key], style.SCOPE_COLOR[SCOPE_LABELS[key]], floor=axis.low)
        medians[key] = binning.median(values)
    smaller, larger = sorted(medians, key=medians.get)
    for key, offset, align in ((smaller, (-12, 27), "right"), (larger, (12, -27), "left")):
        color = style.SCOPE_COLOR[SCOPE_LABELS[key]]
        marks.median_label(plot, medians[key], color, axis.median_text(medians[key]), offset, align)
    medians["ratio"] = median_ratio(plot, medians["human"], medians["benchmark"])
    axes.apply(plot, "x", axis.quantity, axis.noun, low=axis.low, high=axis.high)
    axes.apply(plot, "y", "cdf", axis.y_noun)
    marks.legend_inside(plot, "upper left")
    style.save(figure, output_directory, name)
    return medians


def as_numbers(medians: dict[str, float]) -> dict:
    """The two medians and their ratio, as the figure's numbers."""
    return {
        "median_human": medians["human"],
        "median_benchmark": medians["benchmark"],
        "ratio": medians["ratio"],
    }


def turn_duration(human: reader.Release, benchmark: reader.Release, output_directory: pathlib.Path) -> dict:
    """CDFs of how long one user turn lasts, human-driven against benchmark."""
    axis = ComparisonAxis("duration", "turn duration", MIN_DURATION_DRAWN_S, 86_400, marks.duration_text, "turns")
    medians = draw_comparison(turns.turn_durations(human.requests), turns.turn_durations(benchmark.requests),
                              axis, "turn_duration", output_directory)
    return as_numbers(medians)


def session_duration(human: reader.Release, benchmark: reader.Release, output_directory: pathlib.Path) -> dict:
    """CDFs of how long a whole session lasts, human-driven against benchmark."""
    axis = ComparisonAxis("duration", "session duration", MIN_DURATION_DRAWN_S, 604_800, marks.duration_text,
                          "sessions")
    medians = draw_comparison(turns.session_durations(human.requests), turns.session_durations(benchmark.requests),
                              axis, "session_duration", output_directory)
    return as_numbers(medians)


def input_per_request(human: reader.Release, benchmark: reader.Release, output_directory: pathlib.Path) -> dict:
    """CDFs of the input tokens of one request, human-driven against benchmark."""
    axis = ComparisonAxis("tokens", "input tokens", MIN_INPUT_DRAWN, 1_000_000, marks.count_text, "requests")
    medians = draw_comparison(human.requests.prompt_tokens.dropna(), benchmark.requests.prompt_tokens.dropna(),
                              axis, "input_per_request", output_directory)
    return as_numbers(medians)


def user_prompt_size(human: reader.Release, benchmark: reader.Release, output_directory: pathlib.Path) -> dict:
    """CDFs of the tokens of a user's new prompt, human-driven against benchmark."""
    axis = ComparisonAxis("tokens", "user prompt tokens", MIN_PROMPT_DRAWN, 100_000, marks.count_text, "prompts")
    medians = draw_comparison(tokens.user_prompt_tokens(human.requests), tokens.user_prompt_tokens(benchmark.requests),
                              axis, "user_prompt_size",
                              output_directory)
    return as_numbers(medians)
