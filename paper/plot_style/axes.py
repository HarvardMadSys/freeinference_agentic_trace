"""Axes by quantity: each quantity has one label form, scale and set of ticks.

A figure names the quantity and a noun, such as `apply(axes, "x", "duration",
"session duration")`, so two panels of one quantity read the same way; a
figure labels an axis itself only when no quantity covers it.
"""

import dataclasses
import math

import numpy
from matplotlib import ticker

from paper.plot_style import style

# Durations are labelled with times a person says out loud.
DURATION_TICKS = {1: "1s", 60: "1m", 3600: "1h", 86400: "1d", 604800: "7d"}
LATENCY_TICKS = {0.1: "100ms", 1: "1s", 10: "10s", 60: "1m"}
LATENCY_MS_TICKS = {100.0: "100", 1e3: "1k", 1e4: "10k", 1e5: "100k", 1e6: "1M", 1e7: "10M"}  # a tool's latency in ms
SHARE_TICKS = {0.0: "0", 0.2: "0.2", 0.4: "0.4", 0.6: "0.6", 0.8: "0.8", 1.0: "1"}
TENTH_TICKS = {tenth / 10: f"{tenth / 10:g}" for tenth in range(11)}  # a ratio axis that shows only its top
PERCENTILE_TICKS = {0: "0", 25: "25", 50: "50", 75: "75", 100: "100"}
SECONDS_TICKS = {float(seconds): f"{seconds}" for seconds in range(0, 301, 10)}  # a linear axis of seconds
MEAN_CALLS_TICKS = {1 + fifth / 5: f"{1 + fifth / 5:g}" for fifth in range(0, 11)}  # 1, 1.2, ... 3
TRAJECTORY_TICKS = {20.0: "20", 40.0: "40", 60.0: "60", 80.0: "80", 100.0: "100"}
SPEEDUP_STEP = 0.2  # the speedup axis's ticks: 1×, 1.2×, 1.4×, ...
MAX_WHOLE_MULTIPLE_TICKS = 6  # a whole-multiple axis has at most about this many ticks: 0×, 1×, 2×, ... or 0×, 5×, 10×, ...
NO_TTL_POSITION = 864_000  # "no TTL" is drawn at ten days, one step past a day, and labelled ∞
TTL_TICKS = {1: "1s", 60: "1m", 900: "15m", 86_400: "1d", NO_TTL_POSITION: "∞"}
GAP_TICKS = {10: "10s", 60: "1m", 300: "5m", 900: "15m", 3_600: "1h"}  # gaps up to an hour
MULTIPLE_TICKS = {1: "1×", 2: "2×", 5: "5×", 10: "10×", 20: "20×", 50: "50×"}  # a log axis of multiples
NEAR_ONE_TICKS = {tenth / 10: f"{tenth / 10:g}×" for tenth in range(31)}  # a linear axis of multiples near 1×: 0.9×, 1×
MINOR_PER_MAJOR = 2  # a linear axis has one minor tick between two major ones
LOG_MINOR_SUBS = (2, 3, 4, 5, 6, 7, 8, 9)  # a log axis has a minor tick at each whole multiple of a decade

# Words that stay lower case inside a label.
MINOR_WORDS = frozenset({"a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "by", "for", "per", "vs"})


@dataclasses.dataclass(frozen=True)
class Quantity:
    """How one quantity's axis looks: its label form, scale and ticks."""

    label: str  # `{noun}` is filled in by the figure
    scale: str  # linear or log
    ticks: dict | None = None  # fixed ticks; None for decades on a log scale
    kilo: bool = False  # label decades as 1k, 1M
    window: tuple[float, float] | None = None  # the fixed axis window; None when the figure gives it
    # A time axis names human times and nothing else: a minor rule between 1s and 1m has no name a reader
    # can say, so time axes draw no minor ticks and no minor grid.
    minors: bool = True


QUANTITIES = {
    "cdf": Quantity("{noun} (CDF)", "linear", SHARE_TICKS, window=(0.0, 1.0)),
    "share": Quantity("{noun} (Share)", "linear", SHARE_TICKS, window=(0.0, 1.0)),
    "ratio": Quantity("{noun}", "linear", SHARE_TICKS, window=(0.0, 1.0)),
    "ratio_top": Quantity("{noun}", "linear", TENTH_TICKS),
    "count": Quantity("{noun}", "log"),
    "tokens": Quantity("{noun}", "log", kilo=True),
    "duration": Quantity("{noun}", "log", DURATION_TICKS, minors=False),
    "latency": Quantity("{noun}", "log", LATENCY_TICKS, minors=False),
    "latency_ms": Quantity("{noun} (ms)", "log", LATENCY_MS_TICKS),
    "percentile": Quantity("session percentile", "linear", PERCENTILE_TICKS, window=(0.0, 100.0)),
    "speedup": Quantity("{noun}", "linear"),
    "ttl": Quantity("{noun}", "log", TTL_TICKS, minors=False),
    "gap": Quantity("{noun}", "log", GAP_TICKS, minors=False),
    "multiple": Quantity("{noun}", "log", MULTIPLE_TICKS),
    "multiple_near_one": Quantity("{noun}", "linear", NEAR_ONE_TICKS),
    "whole_multiple": Quantity("{noun}", "linear"),  # 0×, 1×, 2×, ...
    "seconds": Quantity("{noun} (s)", "linear", SECONDS_TICKS),
    "mean_calls": Quantity("{noun}", "linear", MEAN_CALLS_TICKS),
    "trajectory": Quantity("{noun} (%)", "linear", TRAJECTORY_TICKS),
}


def title_case(text: str) -> str:
    """Title Case for a label; minor words stay lower case, words with capitals stay as written."""
    words = text.split()
    cased = []
    for position, word in enumerate(words):
        if (position > 0 and word in MINOR_WORDS) or any(letter.isupper() for letter in word):
            cased.append(word)
        else:
            cased.append(word[:1].upper() + word[1:])
    return " ".join(cased)


def kilo(value: float) -> str:
    """A tick value as 10, 1k, 100k, 1M."""
    if abs(value) >= 1e6:
        return f"{value / 1e6:g}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:g}k"
    return f"{value:g}"


MAX_DECADE_LABELS = 6  # a log axis spanning more decades labels every other one, so its labels never touch


def decade_ticks(low: float, high: float, as_kilo: bool) -> dict:
    """The powers of ten from `low` to `high`, labelled; every other one when there are more than six."""
    exponents = numpy.arange(numpy.ceil(numpy.log10(low)), numpy.floor(numpy.log10(high)) + 1)
    if len(exponents) > MAX_DECADE_LABELS:
        exponents = exponents[exponents % 2 == exponents[-1] % 2]
    return {10.0**exponent: kilo(10.0**exponent) if as_kilo else f"{10.0**exponent:g}" for exponent in exponents}


def apply(axes, which: str, quantity: str, noun: str = "", low: float | None = None, high: float | None = None) -> None:
    """Give one axis its quantity's label, scale, window and ticks."""
    spec = QUANTITIES[quantity]
    axis = axes.xaxis if which == "x" else axes.yaxis
    if spec.scale == "log":
        (axes.set_xscale if which == "x" else axes.set_yscale)("log")
    current_low, current_high = spec.window or (axes.get_xlim() if which == "x" else axes.get_ylim())
    low = current_low if low is None else low
    high = current_high if high is None else high
    if quantity == "speedup":
        positions = numpy.arange(1.0, high + 1e-9, SPEEDUP_STEP)
        ticks = {position: f"{position:g}{style.TIMES}" for position in positions}
    elif quantity == "whole_multiple":
        step = max(1, math.ceil(high / MAX_WHOLE_MULTIPLE_TICKS))
        ticks = {float(position): f"{position}{style.TIMES}" for position in range(0, int(high) + 1, step)}
    elif spec.ticks is None:
        ticks = decade_ticks(low, high, spec.kilo)
    else:
        ticks = {position: text for position, text in spec.ticks.items() if low / 1.001 <= position <= high * 1.001}
    axis.set_major_locator(ticker.FixedLocator(list(ticks)))
    axis.set_major_formatter(ticker.FixedFormatter(list(ticks.values())))
    if spec.minors:
        axis.set_minor_locator(minor_locator(spec.scale))
        axes.grid(True, which="minor", axis=which, alpha=style.MINOR_GRID_ALPHA)
    else:
        axis.set_minor_locator(ticker.NullLocator())
    (axes.set_xlim if which == "x" else axes.set_ylim)(low, high)
    label = title_case(spec.label.replace("{noun}", noun).strip())
    (axes.set_xlabel if which == "x" else axes.set_ylabel)(label)


def minor_locator(scale: str) -> ticker.Locator:
    """Minor ticks between the major ones: whole multiples of a decade on a log axis, halves on a linear one."""
    if scale == "log":
        return ticker.LogLocator(subs=LOG_MINOR_SUBS, numticks=100)
    return ticker.AutoMinorLocator(MINOR_PER_MAJOR)


def category_axis(axes, which: str, positions: list, names: list[str]) -> None:
    """A categorical axis: the names are the ticks."""
    axis = axes.xaxis if which == "x" else axes.yaxis
    axis.set_major_locator(ticker.FixedLocator(positions))
    axis.set_major_formatter(ticker.FixedFormatter(names))
    axis.set_minor_locator(ticker.NullLocator())
    for label in axis.get_ticklabels():
        label.set_fontsize(style.FONT_CATEGORY)
    axes.grid(False, axis=which)
