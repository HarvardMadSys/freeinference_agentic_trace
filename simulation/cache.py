"""The prefix cache, replayed on one partition of a week's reads.

A request reads its blocks when it arrives and writes them when it completes.
Reads of a block are taken in order of arrival, and each is judged against
the block's previous read. With `gap` the arrival minus the previous read's
completion, a read is a hit when the gap is at most the TTL and no eviction
signal removed the block first, or when the gap is negative: the previous
request is still in flight and will write the block. A block's first read of
the week is a miss.

After each read's write, the block stays in the cache until the next read's
completion when that read hits, and otherwise until it leaves: at the TTL, at
an eviction signal, or at the end of the week. That time is reusable when the
block is read again later in the week, whether or not that read hits, and dead
when the block is never read again.

Eviction signals, per policy:
- `ttl_only`: none;
- `reclaim_mutate`: the session's next request arrives without the block;
- `reclaim_dead`: the session's last request completes;
- `reclaim_both`: either.
"""

import dataclasses

import numpy
import pandas

from simulation import block_reads

BLOCK_TOKENS = 16
TTLS = (
    ("1s", 1), ("10s", 10), ("30s", 30), ("1m", 60), ("5m", 300), ("10m", 600), ("15m", 900),
    ("30m", 1_800), ("1h", 3_600), ("3h", 10_800), ("6h", 21_600), ("1d", 86_400), ("none", numpy.inf),
)
POLICIES = ("ttl_only", "reclaim_mutate", "reclaim_dead", "reclaim_both")


@dataclasses.dataclass(frozen=True)
class SortedReads:
    """One partition's reads in order of block then arrival; per read, the arrays below."""

    request: numpy.ndarray  # the request's position in the week
    completion_s: numpy.ndarray
    first: numpy.ndarray  # whether this is the block's first read of the week
    has_next: numpy.ndarray  # whether the block is read again after this read
    gap_to_next_s: numpy.ndarray  # the next read's arrival minus this read's completion; NaN without one
    next_completion_s: numpy.ndarray  # the next read's completion; NaN without one
    signal_s: dict  # per policy, when a signal removes the block after this read; infinite when none does


def kept_by_next_step(hashes: numpy.ndarray, request: numpy.ndarray, week: block_reads.WeekRequests) -> numpy.ndarray:
    """Whether the session's next request reads the same block again."""
    frame = pandas.DataFrame({"block": hashes, "session": week.session[request], "step": week.step[request]})
    # Each read, moved one step back, marks the read of the step before it that it keeps; a left merge against those
    # marks answers the question for every read at once, where a set of (block, session, step) tuples would be far
    # slower on a partition of tens of millions of reads.
    following = frame.assign(step=frame.step - 1).drop_duplicates()
    matched = frame.merge(following, on=["block", "session", "step"], how="left", indicator=True)
    return (matched["_merge"] == "both").to_numpy()


def next_read(values: numpy.ndarray) -> numpy.ndarray:
    """Each read's value taken from the read after it in the sorted order; NaN for the last read."""
    shifted = numpy.full(len(values), numpy.nan)
    shifted[:-1] = values[1:]
    return shifted


def sort_reads(hashes: numpy.ndarray, request: numpy.ndarray, week: block_reads.WeekRequests) -> SortedReads:
    """The partition's reads, sorted, with each read's gap to the next and its eviction signals."""
    arrival = week.arrival_s[request]
    order = numpy.lexsort((arrival, hashes))
    hashes, request, arrival = hashes[order], request[order], arrival[order]
    completion = week.completion_s[request]
    count = len(hashes)
    same_as_next = numpy.zeros(count, dtype=bool)  # whether the read after this one reads the same block
    same_as_next[:-1] = hashes[1:] == hashes[:-1]
    gap = numpy.where(same_as_next, next_read(arrival) - completion, numpy.nan)
    first = numpy.ones(count, dtype=bool)  # a read is a block's first unless the read before it shares the block
    first[1:] = ~same_as_next[:-1]
    kept = kept_by_next_step(hashes, request, week)
    mutate_signal = numpy.where(kept, numpy.inf, week.next_step_arrival_s[request])
    dead_signal = week.session_end_s[request]
    return SortedReads(
        request=request,
        completion_s=completion,
        first=first,
        has_next=same_as_next,
        gap_to_next_s=gap,
        next_completion_s=numpy.where(same_as_next, next_read(completion), numpy.nan),
        signal_s={
            "ttl_only": numpy.full(len(hashes), numpy.inf),
            "reclaim_mutate": mutate_signal,
            "reclaim_dead": dead_signal,
            "reclaim_both": numpy.minimum(mutate_signal, dead_signal),
        },
    )


def tally(sorted_reads: SortedReads, ttl_s: float, policy: str, week_end_s: float) -> dict:
    """Hit and missed tokens, and reusable and dead token-seconds, under one TTL and policy."""
    gap = sorted_reads.gap_to_next_s
    has_next = sorted_reads.has_next
    signal_after = sorted_reads.signal_s[policy] - sorted_reads.completion_s  # seconds after this read's completion
    # The next read hits when the previous request is still in flight and will write the block, or when the gap is at
    # most the TTL and no eviction signal removed the block first. Without a next read nothing hits (the gap is NaN).
    in_flight = has_next & (gap < 0)
    within_ttl = has_next & (gap <= ttl_s)
    before_signal = gap <= signal_after
    next_hits = in_flight | (within_ttl & before_signal)
    # After its write the block stays until the next read's completion when that read hits, and otherwise until it
    # leaves: at the TTL, at the eviction signal, or at the end of the week.
    until_it_leaves = numpy.minimum(numpy.minimum(ttl_s, signal_after), week_end_s - sorted_reads.completion_s)
    until_next_completion = sorted_reads.next_completion_s - sorted_reads.completion_s
    stays_s = numpy.maximum(numpy.where(next_hits, until_next_completion, until_it_leaves), 0)
    # That time is reusable when the block is read again later in the week, and dead when it never is.
    reusable_s = numpy.where(has_next, stays_s, 0.0)
    dead_s = numpy.where(has_next, 0.0, stays_s)
    hits = int(next_hits.sum())
    return {
        "hit_tokens": BLOCK_TOKENS * hits,
        "miss_tokens": BLOCK_TOKENS * (len(gap) - hits),
        "reusable_token_seconds": BLOCK_TOKENS * float(reusable_s.sum()),
        "dead_token_seconds": BLOCK_TOKENS * float(dead_s.sum()),
    }
