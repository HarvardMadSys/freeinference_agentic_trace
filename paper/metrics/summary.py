"""The trace summary: the population's totals, and its sessions and input tokens per harness.

It counts the whole population, SWE-Bench included, with SWE-Bench as its own
harness row. Token totals are the providers' counts; a missing count adds 0.
"""

import pandas

from release import population, reader

DAYS_PER_MONTH = 30.44
MILLISECONDS_PER_DAY = 86_400_000


def totals(everything: reader.Release) -> dict:
    """The trace's totals: its span in months, sessions, requests, tool calls, tokens, and harnesses besides the
    benchmark's."""
    requests = everything.requests
    span_days = (requests.end_ms.max() - requests.start_ms.min()) / MILLISECONDS_PER_DAY
    harnesses = set(everything.sessions.harness) - {population.BENCHMARK_HARNESS}
    return {
        "months": round(span_days / DAYS_PER_MONTH, 1),
        "sessions": int(len(everything.sessions)),
        "requests": int(len(requests)),
        "tool_calls": int(requests.n_tool_calls.sum()),
        "input_tokens": float(requests.prompt_tokens.fillna(0).sum()),
        "output_tokens": float(requests.completion_tokens.fillna(0).sum()),
        "cached_tokens": float(requests.cache_read_tokens.fillna(0).sum()),
        "harnesses": len(harnesses),
    }


def by_harness(everything: reader.Release) -> pandas.DataFrame:
    """Per harness: sessions and input tokens, the largest first."""
    sessions = everything.sessions.groupby("harness").size().rename("sessions")
    tokens = everything.requests.groupby("harness").prompt_tokens.sum().rename("input_tokens")
    table = pandas.concat([sessions, tokens], axis=1).fillna(0)
    return table.sort_values("sessions", ascending=False)
