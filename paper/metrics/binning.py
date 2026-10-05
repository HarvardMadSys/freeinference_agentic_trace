"""Empirical CDFs and medians, rank bins over sessions, and profiles over size bins: the binned forms the paper's
figures report."""

import numpy
import pandas

MAX_RANK_BINS = 200  # a rank-binned stack never has more bins than this
SESSIONS_PER_RANK_BIN = 10  # nor fewer sessions per bin than this
SIZE_BIN_EDGES = 22  # geometric edges from 1 to the 99th percentile of x
MIN_ROWS_PER_SIZE_BIN = 50  # a size bin with fewer rows is not drawn


def cdf(values: pandas.Series) -> tuple[numpy.ndarray, numpy.ndarray]:
    """The sorted finite values, and the share of values at or below each."""
    sorted_values = numpy.sort(numpy.asarray(values, dtype=float))
    sorted_values = sorted_values[numpy.isfinite(sorted_values)]
    return sorted_values, numpy.arange(1, len(sorted_values) + 1) / len(sorted_values)


def median(values: pandas.Series) -> float:
    """The median of the finite values."""
    return float(numpy.nanmedian(numpy.asarray(values, dtype=float)))


def rank_bins(shares: pandas.DataFrame, sort_by: list[str]) -> pandas.DataFrame:
    """Sessions sorted by `sort_by`, cut into equal bins, and each bin's mean shares.

    The result has one row per bin, its `percentile` and one column per share;
    each row sums to 1.
    """
    ordered = shares.sort_values(sort_by, kind="stable").reset_index(drop=True)
    bin_count = min(MAX_RANK_BINS, max(len(ordered) // SESSIONS_PER_RANK_BIN, 1))
    bins = (numpy.arange(len(ordered)) * bin_count) // len(ordered)
    means = ordered.groupby(bins).mean()
    means = means.div(means.sum(axis=1), axis=0)
    means["percentile"] = (means.index + 0.5) * 100 / bin_count
    return means


def size_bins(x: pandas.Series, values: pandas.DataFrame) -> pandas.DataFrame:
    """Per bin of x and per value column: the 25th, 50th and 75th percentiles and the row count.

    Bins have geometric edges from 1 to the 99th percentile of x, right-closed;
    a bin with fewer than 50 rows is left out. `mid` is the bin's midpoint.
    """
    upper = float(numpy.percentile(x, 99))
    edges = numpy.unique(numpy.round(numpy.geomspace(1, upper, SIZE_BIN_EDGES)))
    intervals = pandas.cut(x, edges, include_lowest=True)
    rows = []
    for column in values.columns:
        grouped = values[column].groupby(intervals, observed=True)
        for interval, group in grouped:
            if len(group) < MIN_ROWS_PER_SIZE_BIN:
                continue
            rows.append(
                {
                    "series": column,
                    "mid": interval.mid,
                    "p25": group.quantile(0.25),
                    "p50": group.quantile(0.5),
                    "p75": group.quantile(0.75),
                    "n": len(group),
                }
            )
    return pandas.DataFrame(rows)
