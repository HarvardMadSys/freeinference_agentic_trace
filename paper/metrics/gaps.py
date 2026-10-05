"""The gap before each request, what it waited for, and the cache hit ratio after it.

A request's gap is its arrival minus the previous request's completion in the
same session; a session's first request has none. Its cause is what the
request waited for, named as the paper names it. Its hit ratio is the share of
its input the provider served from cache.

A request's idle gap at its provider is its arrival minus the latest
completion among the same user's earlier requests at that provider, in any
session: how long that provider's cache went unused by the user.
"""

import math

import numpy
import pandas

from release import schema
from paper.metrics import tokens, turns

# What a request waited for, named as the paper names the gap before it, in the paper's order.
CAUSE_NAMES = {"tool": "Tool Gap", "ask_user_tool": "Ask-User Tool", "user": "User Idle", "subagent": "Subagent"}

# The gap bins: (name, label, lower bound, upper bound) in seconds, lower bound included.
GAP_BINS = (
    ("under_1m", "<1m", 0, 60),
    ("1m_to_5m", "1–5m", 60, 300),
    ("5m_to_15m", "5–15m", 300, 900),
    ("15m_to_30m", "15–30m", 900, 1_800),
    ("30m_to_1h", "30m–1h", 1_800, 3_600),
    ("over_1h", ">1h", 3_600, math.inf),
)

# The idle-gap bins: 26 bins from 10 s to 1 h, each 1.25 times as wide as the one
# before, fine enough to place a provider's cliff within a few minutes.
IDLE_GAP_EDGES_S = tuple(numpy.geomspace(10.0, 3_600.0, 27))
MIN_ROWS_PER_IDLE_GAP_GROUP = 20  # a thinner bin is merged into the next, so no point rests on a handful of requests


def hit_ratio(requests: pandas.DataFrame) -> pandas.Series:
    """Cached over input tokens, at most 1, on cache-reporting rows with an input; missing otherwise."""
    usable = tokens.is_cache_reporting(requests) & (requests.prompt_tokens > 0)
    ratio = (requests.cache_read_tokens / requests.prompt_tokens).clip(upper=1.0)
    return ratio.where(usable)


def gap_bin(gap_s: pandas.Series) -> pandas.Series:
    """Each gap's bin label."""
    edges = [lower for _, _, lower, _ in GAP_BINS] + [math.inf]
    labels = [label for _, label, _, _ in GAP_BINS]
    return pandas.cut(gap_s, bins=edges, labels=labels, right=False).astype(str)


def gap_rows(requests: pandas.DataFrame) -> pandas.DataFrame:
    """One row per request with a gap of 0 s or more and a cause: the gap, its cause and bin, and the hit ratio.

    The hit ratio is missing where the provider reported no cache reads.
    Requests must be in session and step order.
    """
    gaps = pandas.DataFrame(
        {
            "gap_s": turns.waits(requests),
            "cause": requests.waited_for.map(CAUSE_NAMES),
            "hit_ratio": hit_ratio(requests),
        }
    )
    gaps = gaps[gaps.gap_s.notna() & (gaps.gap_s >= 0) & gaps.cause.notna()].copy()
    gaps["gap_bin"] = gap_bin(gaps.gap_s)
    return gaps


def idle_gap_rows(requests: pandas.DataFrame) -> pandas.DataFrame:
    """Per request with a hit ratio whose user's previous request at the provider also reported cache reads:
    its provider, idle gap and hit ratio.

    Every earlier request counts towards the latest completion, reported or not,
    since serving a request warms the cache. A request after one that did not
    report is dropped: its gap reaches back past a request that already warmed
    the cache.
    """
    ordered = requests[requests.provider.notna()].sort_values(["provider", "user_id", "start_ms"], kind="mergesort")
    keys = [ordered.provider, ordered.user_id]
    latest_earlier_completion_ms = ordered.groupby(keys).end_ms.cummax().groupby(keys).shift()
    previous_reported = ordered.groupby(keys).cache_read_tokens.shift().notna()
    rows = pandas.DataFrame(
        {
            "provider": ordered.provider,
            "idle_gap_s": (ordered.start_ms - latest_earlier_completion_ms) / schema.MILLISECONDS_PER_SECOND,
            "hit_ratio": hit_ratio(ordered),
        }
    )
    return rows[rows.idle_gap_s.notna() & rows.hit_ratio.notna() & previous_reported]


def idle_gap_groups(rows: pandas.DataFrame) -> list[pandas.DataFrame]:
    """The rows in idle-gap bins, a bin with too few rows merged into the next one.

    Gaps over an hour are dropped; gaps under 10 s, in-flight ones included,
    count as 10 s. A thin tail joins the last group rather than being dropped,
    so a curve reaches the end of the axis.
    """
    rows = rows[rows.idle_gap_s <= IDLE_GAP_EDGES_S[-1]].copy()
    rows["idle_gap_s"] = rows.idle_gap_s.clip(lower=IDLE_GAP_EDGES_S[0])
    bins = pandas.cut(rows.idle_gap_s, IDLE_GAP_EDGES_S, include_lowest=True)
    groups, carried = [], []
    for _, in_bin in rows.groupby(bins, observed=False):
        carried.append(in_bin)
        if sum(len(part) for part in carried) >= MIN_ROWS_PER_IDLE_GAP_GROUP:
            groups.append(pandas.concat(carried))
            carried = []
    leftover = [part for part in carried if len(part)]
    if leftover and groups:
        groups[-1] = pandas.concat([groups[-1], *leftover])
    return groups


def retention_points(rows: pandas.DataFrame) -> pandas.DataFrame:
    """Per idle-gap group: the median gap of its rows, their median hit ratio, and how many rows it holds."""
    groups = idle_gap_groups(rows)
    return pandas.DataFrame(
        {
            "gap_s": [float(group.idle_gap_s.median()) for group in groups],
            "hit_ratio": [float(group.hit_ratio.median()) for group in groups],
            "rows": [len(group) for group in groups],
        }
    )


def causes_present(rows: pandas.DataFrame) -> list[tuple[str, str]]:
    """The (waited-for value, cause name) pairs that have gap rows, in the paper's order."""
    present = set(rows.cause)
    return [(waited_for, cause) for waited_for, cause in CAUSE_NAMES.items() if cause in present]


def mean_hit_ratio_by_cause(rows: pandas.DataFrame) -> pandas.Series:
    """Per cause present, in the paper's order: the mean hit ratio of the requests whose gap had that cause."""
    rows = rows.dropna(subset=["hit_ratio"])
    causes = [cause for _, cause in causes_present(rows)]
    return pandas.Series([float(rows.hit_ratio[rows.cause == cause].mean()) for cause in causes], index=causes)
