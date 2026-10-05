"""What three or more figures draw with: CDF lines, bands, stacks, bars, median labels, reference lines and legends.

A mark one figure alone uses lives in that figure's module."""

import numpy

from paper.metrics import binning
from paper.plot_style import style


def cdf_line(axes, values, label: str, color: str, floor: float | None = None, dashed: bool = False):
    """An empirical CDF of `values`, each first raised to `floor` when one is given."""
    values = numpy.asarray(values, dtype=float)
    if floor is not None:
        values = numpy.maximum(values, floor)
    x, y = binning.cdf(values)
    return axes.plot(x, y, color=color, lw=style.LINE_WIDTH, ls="--" if dashed else "-", label=label)[0]


def band_line(axes, table, label: str, color: str, window: tuple[float, float], band_floor: float | None = None):
    """A size-binned median line with its 25th to 75th percentile band, held out to the axis window."""
    table = table.sort_values("mid")
    x = numpy.concatenate(([window[0]], table.mid.to_numpy(dtype=float), [window[1]]))
    p25 = table.p25.to_numpy(dtype=float)
    if band_floor is not None:
        p25 = numpy.maximum(p25, band_floor)
    p50 = table.p50.to_numpy(dtype=float)
    p75 = table.p75.to_numpy(dtype=float)
    p25, p50, p75 = (numpy.concatenate((values[:1], values, values[-1:])) for values in (p25, p50, p75))
    axes.fill_between(x, p25, p75, color=color, alpha=0.15, lw=0)
    return axes.plot(x, p50, color=color, lw=style.LINE_WIDTH, label=label)[0]


def stack_area(axes, x, bands: list[tuple[str, numpy.ndarray, str]]):
    """Shares stacked bottom to top; `bands` is a list of (label, values, colour)."""
    labels = [band[0] for band in bands]
    colors = [band[2] for band in bands]
    return axes.stackplot(x, *[band[1] for band in bands], labels=labels, colors=colors, edgecolor="none")


def horizontal_bars(axes, positions, values, colors, left: float, value_labels: list[str]):
    """Bars from `left` to each value, each labelled at its end."""
    axes.barh(positions, numpy.asarray(values) - left, left=left, height=0.55, color=colors, alpha=0.9, zorder=3)
    pad = (max(values) - left) * 0.03
    for position, value, text in zip(positions, values, value_labels):
        axes.text(value + pad, position, text, va="center", ha="left", fontsize=style.FONT_NOTE, color="black")


REFERENCE_DOTS = (0, (1, 2))  # a dotted reference, such as the 1× of a normalised cost
BASELINE_DASHES = (0, (6, 3, 1, 3))  # a dash-dot baseline, such as the active blocks


def reference_line(axes, y: float):
    """A dotted grey line at a reference value."""
    axes.axhline(y, color=style.REFERENCE_GREY, lw=1.0, ls=REFERENCE_DOTS, zorder=1)


def baseline(axes, y: float, label: str | None = None):
    """A dark dash-dot line at a baseline, with its label above it at the left when given."""
    axes.axhline(y, color=style.GREY, lw=1.4, ls=BASELINE_DASHES, zorder=1)
    if label:
        axes.annotate(label, (0.04, y), xycoords=("axes fraction", "data"), xytext=(0, 4), textcoords="offset points",
                      ha="left", va="bottom", fontsize=style.FONT_CALLOUT, color=style.INK)


def duration_text(seconds: float) -> str:
    """A duration as the paper prints it: 61.0s, 192s, 30.6m, 450ms."""
    if seconds >= 300:
        return f"{seconds / 60:.1f}m"
    if seconds >= 100:
        return f"{seconds:.0f}s"
    if seconds >= 0.1:
        return f"{seconds:.1f}s"
    return f"{seconds * 1000:.0f}ms"


def count_text(value: float) -> str:
    """A count as the paper prints it: 44, 122.1k."""
    return f"{value / 1000:.1f}k" if value >= 1000 else f"{value:.0f}"


def median_label(axes, value: float, color: str, text: str, offset: tuple[int, int] = (-12, 27), align: str = "right"):
    """A dot on the CDF's median, with an arrow to its value."""
    axes.plot([value], [0.5], marker="o", ms=3.4, color=color, mew=0, ls="none", zorder=8)
    axes.annotate(
        text, (value, 0.5), xytext=offset, textcoords="offset points", ha=align, va="center",
        fontsize=style.FONT_CALLOUT, color=style.INK, zorder=9,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
        arrowprops={"arrowstyle": "->", "color": "0.45", "lw": 0.7, "shrinkA": 1.5, "shrinkB": 3.5},
    )


LEGEND_ABOVE_COLUMNS = 2  # a legend above the plot wraps into two columns, as the paper's do


def legend_above(axes, columns: int = LEGEND_ABOVE_COLUMNS, font_size: float | None = None, compact: bool = False):
    """The legend above the plot, in two columns and at the legend size unless told otherwise; `compact` narrows
    its handles and the gaps between its columns."""
    handles, labels = axes.get_legend_handles_labels()
    spacing = {"handlelength": 0.8, "columnspacing": 0.5, "handletextpad": 0.3} if compact else \
        {"handlelength": 1.2, "columnspacing": 1.0, "handletextpad": 0.4}
    return axes.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=columns, fontsize=font_size,
                       **spacing)


def legend_inside(axes, location: str):
    """The legend inside the plot, at `location`."""
    return axes.legend(loc=location, handlelength=1.2, handletextpad=0.4, labelspacing=0.3)


