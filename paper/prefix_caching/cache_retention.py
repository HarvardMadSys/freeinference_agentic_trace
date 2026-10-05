"""Retention: what each TTL costs in prefill, storage and money, and what eviction signals save.

Every quantity comes from `metrics.cache_costs.curve`, over the simulation's
tallies summed across the weeks.
"""

import math
import pathlib

import numpy
import pandas

from paper.plot_style import axes, marks, style
from paper.metrics import cache_costs
from simulation import cache

TTL_POSITIONS = {name: (axes.NO_TTL_POSITION if name == "none" else seconds) for name, seconds in cache.TTLS}
TTL_LABELS = {"none": "∞"}  # a TTL's label beside its point; the others use their own names
NO_TTL = "none"
PREFILL_TTLS = ("15m", "30m", "1h", "6h", "1d")  # the TTLs whose prefill above the floor the text quotes
RETAINED_TTLS = ("1h", "1d")  # the TTLs whose retained tokens the text compares with a 15-minute TTL's
ONE_DAY_S = 86_400
# The TTLs labelled on a prefill curve, each at its offset in points from its point, fanned into the open space
# above and to the right of a falling curve; a thin leader joins each label to its point.
HUMAN_TTL_LABELS = {"1s": (17, -9), "10s": (20, 16), "1m": (23, 26), "5m": (29, 34), "15m": (40, 28), "1h": (45, 19),
                    "1d": (21, 12), NO_TTL: (-7, 19)}
HUMAN_TTL_LABELS_BESIDE_BENCHMARK = {"1s": (-26, 0), "10s": (24, 2), "1m": (23, 26), "15m": (40, 28), "1d": (14, 9),
                                     NO_TTL: (-7, 24)}
BENCHMARK_TTL_LABELS = {"1s": (16, 0), "10s": (18, 8), "1m": (34, 25), "5m": (32, 12)}
LEADER = {"arrowstyle": "-", "color": "0.45", "lw": 0.7, "shrinkA": 1.5, "shrinkB": 3.5}
POLICIES_TO_NO_TTL = ("reclaim_dead", "reclaim_both")  # without a TTL, only a session-end signal ever frees a block
ACTIVE_LABEL = "= Active"


# ---- a callout on one point of a cost curve
CALLOUT_RISE_SHARE = 0.45  # a callout's text sits this share of the plot's height above its point


def callout(axes, x: float, y: float, text: str):
    """A hollow dot on the point and a straight vertical leader up to its text."""
    axes.plot([x], [y], marker="o", ms=6, mfc="white", mec=style.INK, mew=1.6, ls="none", zorder=7, clip_on=False)
    rise = axes.get_window_extent().height * 72 / axes.figure.dpi * CALLOUT_RISE_SHARE
    axes.annotate(text, (x, y), xytext=(0, rise), textcoords="offset points", ha="center", fontsize=style.FONT_CALLOUT,
                  color=style.INK, zorder=6, arrowprops={"arrowstyle": "-", "color": style.INK, "lw": 1.2})


def ttl_positions(curve: pandas.DataFrame) -> numpy.ndarray:
    """Where each TTL of a curve is drawn on the TTL axis."""
    return numpy.array([TTL_POSITIONS[name] for name in curve.index], dtype=float)


def finite(curve: pandas.DataFrame) -> pandas.DataFrame:
    """A curve's finite TTLs, up to a day."""
    return curve.drop(index=NO_TTL)


FALLBACK_WINDOW = (1e-3, 1e3)  # the window of a ratio axis with no positive value to fit


def decade_window(values) -> tuple[float, float]:
    """The powers of ten just below the smallest positive value and just above the largest."""
    positive = [float(value) for value in values if value > 0]
    if not positive:
        return FALLBACK_WINDOW
    return 10.0 ** math.floor(math.log10(min(positive))), 10.0 ** math.ceil(math.log10(max(positive)))


def drawn_values(curves: dict[str, pandas.DataFrame], column: str) -> list[float]:
    """Every value of a column the policy figure draws: to no TTL for the policies drawn there, else to a day."""
    values = []
    for policy, curve in curves.items():
        drawn = curve if policy in POLICIES_TO_NO_TTL else finite(curve)
        values.extend(float(value) for value in drawn[column])
    return values


def prefill_against_storage(curves: dict[str, tuple[pandas.DataFrame, str, dict]], output_directory: pathlib.Path,
                            name: str) -> None:
    """Normalised prefill against mean retained tokens, one point per TTL; one line per (label, colour, labelled
    TTLs) curve, each labelled TTL joined to its label by a leader.

    A single curve is drawn with hollow points and no legend.
    """
    figure, plot = style.new_figure()
    single = len(curves) == 1
    for label, (curve, color, labelled) in curves.items():
        plot.plot(curve.mean_retained_tokens, curve.normalised_prefill, marker="o", ms=4.5 if single else 3.5,
                  mfc="white" if single else color, color=color, lw=style.LINE_WIDTH, label=label, zorder=4)
        for ttl, offset in labelled.items():
            plot.annotate(TTL_LABELS.get(ttl, ttl), (curve.mean_retained_tokens[ttl], curve.normalised_prefill[ttl]),
                          xytext=offset, textcoords="offset points", ha="left" if offset[0] >= 0 else "right",
                          va="center", fontsize=style.FONT_NOTE, color=style.INK, arrowprops=LEADER, zorder=7)
    low = min(curve.mean_retained_tokens.min() for curve, _, _ in curves.values()) / 2
    high = max(curve.mean_retained_tokens.max() for curve, _, _ in curves.values()) * 3
    axes.apply(plot, "x", "tokens", "mean retained tokens", low=low, high=high)
    axes.apply(plot, "y", "multiple", "normalized prefill", low=0.9, high=60)
    if len(curves) > 1:
        marks.legend_inside(plot, "upper right")
    style.save(figure, output_directory, name)


def retained_over_active(curve: pandas.DataFrame, output_directory: pathlib.Path) -> None:
    """Reusable and dead retained tokens over active ones, per TTL up to a day."""
    curve = finite(curve)
    figure, plot = style.new_figure()
    positions = ttl_positions(curve)
    plot.plot(positions, curve.reusable_over_active, color=style.RETAINED_COLOR["Reusable"], lw=style.LINE_WIDTH, label="Reusable")
    plot.plot(positions, curve.dead_over_active, color=style.RETAINED_COLOR["Dead"], lw=style.LINE_WIDTH, label="Dead")
    marks.baseline(plot, 1)
    low, high = decade_window(list(curve.reusable_over_active) + list(curve.dead_over_active))
    axes.apply(plot, "x", "ttl", "cache TTL", low=1, high=ONE_DAY_S)
    axes.apply(plot, "y", "count", "retained / active", low=low, high=high)
    marks.legend_inside(plot, "upper left")
    style.save(figure, output_directory, "retained_over_active")


def stacked_cost(curve: pandas.DataFrame, parts: dict[str, str], colors: dict[str, str], output_directory: pathlib.Path,
                 name: str, mark: tuple[str, str]) -> None:
    """Cost parts stacked per TTL up to a day, over the total at a 1-minute TTL (dotted at 1×), with one TTL called
    out."""
    curve = finite(curve)
    figure, plot = style.new_figure(top_legend=True)
    positions = ttl_positions(curve)
    bands = [(label, curve[column].to_numpy(), colors[label]) for label, column in parts.items()]
    marks.stack_area(plot, positions, bands)
    marks.reference_line(plot, 1.0)
    high = float(curve[list(parts.values())].sum(axis=1).max()) * 1.1
    axes.apply(plot, "x", "ttl", "cache TTL", low=1, high=ONE_DAY_S)
    axes.apply(plot, "y", "whole_multiple", "normalized cost", low=0, high=high)
    marks.legend_above(plot, columns=len(parts))
    figure.canvas.draw()  # the callout's rise is a share of the plot's drawn height
    marked_ttl, text = mark
    callout(plot, TTL_POSITIONS[marked_ttl], float(curve.loc[marked_ttl, list(parts.values())].sum()), text)
    style.save(figure, output_directory, name)


def api_cost(curve: pandas.DataFrame, output_directory: pathlib.Path) -> dict:
    """API cost per TTL, and its knee: the shortest TTL within 5 % of the lowest total."""
    totals = curve.api_total
    knee = cache_costs.api_cost_knee(curve)
    parts = {"Output": "api_output", "Cache Read": "api_cache_read", "Input": "api_input"}
    stacked_cost(curve, parts, style.API_COST_COLOR, output_directory, "api_cost",
                 (knee, f"Knee {totals[knee]:.2f}{style.TIMES} @ {knee}"))
    return {"knee_ttl": knee,
            "knee_cost": float(totals[knee])}


def infrastructure_cost(curve: pandas.DataFrame, output_directory: pathlib.Path) -> dict:
    """Recompute and storage cost per TTL, and the TTL where their total is lowest."""
    totals = curve.infrastructure_total
    lowest = totals.idxmin()
    stacked_cost(curve, {"Recompute": "recompute", "Storage": "storage"}, style.INFRASTRUCTURE_COST_COLOR,
                 output_directory, "infrastructure_cost", (lowest, f"Min {totals[lowest]:.2f}{style.TIMES} @ {lowest}"))
    return {"lowest_ttl": lowest,
            "lowest_cost": float(totals[lowest])}


def by_policy(curves: dict[str, pandas.DataFrame], column: str, output_directory: pathlib.Path, name: str,
              y_quantity: str, y_noun: str, window: tuple[float, float]) -> None:
    """One line per eviction policy of one column of the curves, per TTL; only the policies that free blocks at a
    session's end are drawn out to no TTL."""
    figure, plot = style.new_figure(top_legend=True)
    for policy, curve in curves.items():
        drawn = curve if policy in POLICIES_TO_NO_TTL else finite(curve)
        dashed = "--" if policy == "ttl_only" else "-"
        plot.plot(ttl_positions(drawn), drawn[column], color=style.POLICY_COLOR[policy], lw=style.LINE_WIDTH, ls=dashed,
                  label=style.POLICY_NAMES[policy])
    axes.apply(plot, "x", "ttl", "cache TTL", low=1, high=axes.NO_TTL_POSITION)
    axes.apply(plot, "y", y_quantity, y_noun, low=window[0], high=window[1])
    if y_quantity == "count":
        marks.baseline(plot, 1, ACTIVE_LABEL)
    else:
        marks.reference_line(plot, 1.0)
    marks.legend_above(plot)
    style.save(figure, output_directory, name)
