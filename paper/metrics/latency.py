"""Where a session's time goes: prefill, decode, tools and people.

A request's TTFT is trusted when it is positive and under 95% of its latency;
its prefill is then the TTFT and its decode the rest. The latency of a request
without a trusted TTFT is unsplit, and is shared out by its session's prefill
ratio. Waits are tool time when the request waited for a tool or a subagent,
and human time when it waited for a person, a reply or an answer to a question.
"""

import numpy
import pandas

from release import schema
from paper.metrics import tokens, turns

TRUSTED_TTFT_SHARE = 0.95  # a TTFT at or above 95% of the latency leaves too little decode to trust
TOOL_GAP_CAP_S = 900.0  # capped tool time counts each tool wait for at most 15 minutes
TOOL_WAITS = ("tool", "subagent")
HUMAN_WAITS = ("user", "ask_user_tool")
LONG_MISS_MAX_HIT_RATIO = 0.25  # a long miss reads under a quarter of its input from cache
LONG_MISS_MIN_UNCACHED = 10_000  # and prefills more than 10,000 tokens
MIN_LONG_MISSES = 200  # a model with fewer long misses gets no prefill rate


def is_trusted_ttft(requests: pandas.DataFrame) -> pandas.Series:
    """Whether the TTFT is positive and under 95% of the latency."""
    return (requests.ttft_ms > 0) & (requests.ttft_ms < TRUSTED_TTFT_SHARE * (requests.end_ms - requests.start_ms))


def llm_time(requests: pandas.DataFrame) -> pandas.DataFrame:
    """Per request, in seconds: its prefill and decode when its TTFT is trusted, else its unsplit latency."""
    trusted = is_trusted_ttft(requests).fillna(False)
    latency_s = (requests.end_ms - requests.start_ms) / schema.MILLISECONDS_PER_SECOND
    ttft_s = requests.ttft_ms / schema.MILLISECONDS_PER_SECOND
    return pandas.DataFrame(
        {
            "session_id": requests.session_id,
            "harness": requests.harness,
            "prefill": ttft_s.where(trusted, 0.0),
            "decode": (latency_s - ttft_s).where(trusted, 0.0),
            "unsplit": latency_s.where(~trusted, 0.0),
        }
    )


def effective_llm_time(requests: pandas.DataFrame) -> pandas.DataFrame:
    """Per session: prefill and decode, with its unsplit time shared by its prefill ratio.

    The ratio is the session's own when it has split time, else its harness's
    over these sessions, else the overall one.
    """
    per_request = llm_time(requests)
    sessions = per_request.groupby("session_id").agg(
        harness=("harness", "first"), prefill=("prefill", "sum"), decode=("decode", "sum"), unsplit=("unsplit", "sum")
    )
    split = sessions.prefill + sessions.decode
    by_harness = sessions.groupby("harness")[["prefill", "decode"]].sum()
    harness_split = by_harness.prefill + by_harness.decode
    harness_ratio = (by_harness.prefill / harness_split).where(harness_split > 0)
    overall_split = split.sum()
    overall_ratio = sessions.prefill.sum() / overall_split if overall_split > 0 else 0.0
    ratio = (sessions.prefill / split).where(split > 0)
    ratio = ratio.fillna(sessions.harness.map(harness_ratio)).fillna(overall_ratio)
    return pandas.DataFrame(
        {
            "prefill": sessions.prefill + sessions.unsplit * ratio,
            "decode": sessions.decode + sessions.unsplit * (1 - ratio),
        }
    )


def time_breakdown(requests: pandas.DataFrame, cap_tool_waits: bool = False) -> pandas.DataFrame:
    """Per session, in seconds: effective prefill and decode, tool time and human time."""
    waits = turns.waits(requests).clip(lower=0)
    tool_waits = waits.where(requests.waited_for.isin(TOOL_WAITS), 0.0).fillna(0.0)
    if cap_tool_waits:
        tool_waits = tool_waits.clip(upper=TOOL_GAP_CAP_S)
    human_waits = waits.where(requests.waited_for.isin(HUMAN_WAITS), 0.0).fillna(0.0)
    breakdown = effective_llm_time(requests)
    breakdown["tool"] = tool_waits.groupby(requests.session_id).sum()
    breakdown["human"] = human_waits.groupby(requests.session_id).sum()
    return breakdown


def time_shares(requests: pandas.DataFrame) -> pandas.DataFrame:
    """Per session with any time: the shares of its time that are prefill, decode, tool time and human time."""
    breakdown = time_breakdown(requests)
    breakdown = breakdown[breakdown.sum(axis=1) > 0]
    return breakdown.div(breakdown.sum(axis=1), axis=0)


def speedups(breakdown: pandas.DataFrame) -> dict[str, float]:
    """For prefill, decode and tool time: the session-time speedup if that part alone ran twice as fast.

    Machine time is prefill, decode and tool time; a person's time is not machine time.
    """
    machine = breakdown[["prefill", "decode", "tool"]]
    machine = machine[machine.sum(axis=1) > 0]
    total = machine.to_numpy().sum()
    return {part: float(total / (total - 0.5 * machine[part].sum())) for part in ("prefill", "decode", "tool")}


def ttft_decomposition(requests: pandas.DataFrame, model: str) -> dict | None:
    """For one model, over requests with a trusted TTFT: TTFT, decode and queueing, in seconds.

    The prefill rate is the median of uncached tokens per second of TTFT over
    the model's long misses; queueing is the TTFT minus the uncached tokens at
    that rate. None when the model has too few long misses.
    """
    rows = requests[(requests.model == model) & is_trusted_ttft(requests).fillna(False)]
    uncached = tokens.uncached_tokens(rows)
    ttft_s = rows.ttft_ms / schema.MILLISECONDS_PER_SECOND
    hit_ratio = rows.cache_read_tokens / rows.prompt_tokens
    long_miss = (hit_ratio < LONG_MISS_MAX_HIT_RATIO) & (uncached > LONG_MISS_MIN_UNCACHED)
    if long_miss.sum() < MIN_LONG_MISSES:
        return None
    rate = float(numpy.median(uncached[long_miss] / ttft_s[long_miss]))
    queueing = (ttft_s - uncached / rate).clip(lower=0).dropna()
    return {
        "ttft": ttft_s,
        "decode": (rows.end_ms - rows.start_ms - rows.ttft_ms) / schema.MILLISECONDS_PER_SECOND,
        "queueing": queueing,
        "rate": rate,
        "long_misses": int(long_miss.sum()),
    }
