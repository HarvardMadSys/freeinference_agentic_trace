"""The cost of a cache, per TTL, from the simulation's tallies summed over the weeks: what is retained, what is
prefilled, and what the API and the infrastructure pay.

For one scope and eviction policy, per TTL:
- mean retained tokens: reusable and dead token-seconds over the span;
- normalised prefill: missed tokens over those with no TTL under `ttl_only`;
- reusable and dead over active: their token-seconds over those of blocks held
  by requests in flight;
- API cost: missed input, cache reads and output at list prices, over the
  total at a 1-minute TTL;
- infrastructure cost: recomputing missed tokens and storing retained ones,
  over the total at a 1-minute TTL.

And from the cheaper-reads model, per TTL and read price cut: today's
prefill and API cost as a multiple of those of a harness that mutates less.
"""

import pandas

from simulation import cache
from paper.metrics import prices

NORMALISING_TTL = "1m"  # costs are drawn as a multiple of their total at this TTL
KNEE_TOLERANCE = 0.05  # the knee is the shortest TTL within this share of the lowest API cost
REFERENCE_TTL = "15m"  # retained tokens are compared with those at this TTL


def curve(tallies: pandas.DataFrame, totals: pandas.DataFrame, scope: str, policy: str) -> pandas.DataFrame:
    """One scope's and policy's quantities per TTL, in the order of `cache.TTLS`."""
    order = [name for name, _ in cache.TTLS]
    rows = tallies[(tallies.scope == scope) & (tallies.policy == policy)].set_index("ttl").loc[order]
    floor = tallies[(tallies.scope == scope) & (tallies.policy == "ttl_only") & (tallies.ttl == "none")].miss_tokens.iloc[0]
    scope_totals = totals.loc[scope]
    retained = rows.reusable_token_seconds + rows.dead_token_seconds
    frame = pandas.DataFrame(index=rows.index)
    frame["mean_retained_tokens"] = retained / scope_totals.span_seconds
    frame["normalised_prefill"] = rows.miss_tokens / floor
    frame["reusable_over_active"] = rows.reusable_token_seconds / scope_totals.active_token_seconds
    frame["dead_over_active"] = rows.dead_token_seconds / scope_totals.active_token_seconds
    frame["api_input"] = rows.miss_tokens * prices.UNCACHED_PRICE / prices.TOKENS_PER_MILLION
    frame["api_cache_read"] = rows.hit_tokens * prices.CACHED_PRICE / prices.TOKENS_PER_MILLION
    frame["api_output"] = scope_totals.output_tokens * prices.OUTPUT_PRICE / prices.TOKENS_PER_MILLION
    frame["recompute"] = rows.miss_tokens * prices.RECOMPUTE_PRICE / prices.TOKENS_PER_MILLION
    frame["storage"] = retained / prices.SECONDS_PER_DAY * prices.STORAGE_PRICE_PER_DAY / prices.TOKENS_PER_MILLION
    api_parts = ["api_input", "api_cache_read", "api_output"]
    infrastructure_parts = ["recompute", "storage"]
    api_at_minute = frame.loc[NORMALISING_TTL, api_parts].sum()
    infrastructure_at_minute = frame.loc[NORMALISING_TTL, infrastructure_parts].sum()
    frame[api_parts] = frame[api_parts] / api_at_minute
    frame[infrastructure_parts] = frame[infrastructure_parts] / infrastructure_at_minute
    frame["api_total"] = frame[api_parts].sum(axis=1)
    frame["infrastructure_total"] = frame[infrastructure_parts].sum(axis=1)
    return frame


def cheaper_reads_curves(sums: pandas.DataFrame) -> pandas.DataFrame:
    """Per TTL and read price cut d: today's prefilled tokens and API cost over those of the harness that mutates less.

    A cost is prefilled tokens at the uncached price, cached tokens at (1 - d)
    times the cached price, and output tokens at the output price.
    """
    read_price = (1 - sums.read_price_cut) * prices.CACHED_PRICE
    output_cost = sums.output_tokens * prices.OUTPUT_PRICE
    today_cost = sums.today_prefilled_tokens * prices.UNCACHED_PRICE + sums.today_cached_tokens * read_price + output_cost
    mutates_less_cost = sums.prefilled_tokens * prices.UNCACHED_PRICE + sums.cached_tokens * read_price + output_cost
    curves = sums[["ttl", "read_price_cut"]].copy()
    curves["prefill_multiple"] = sums.today_prefilled_tokens / sums.prefilled_tokens
    curves["cost_multiple"] = today_cost / mutates_less_cost
    return curves


def api_cost_knee(curve: pandas.DataFrame) -> str:
    """The shortest TTL whose API cost is within 5 % of the lowest."""
    totals = curve.api_total
    return next(ttl for ttl in curve.index if totals[ttl] <= totals.min() * (1 + KNEE_TOLERANCE))


def break_even_read_price(curves_of_one_ttl: pandas.DataFrame) -> float | None:
    """The highest read price at and below which today's cost multiple is at least 1; None when the multiple is
    below 1 even at free reads."""
    price = None
    for row in curves_of_one_ttl.sort_values("read_price_cut", ascending=False).itertuples():
        if row.cost_multiple < 1:
            break
        price = round(1 - row.read_price_cut, 2)  # the price cuts are twentieths of $1, so cents are exact
    return price


def retention_summary(curves: dict[str, pandas.DataFrame], ttls_above_floor: tuple,
                      ttls_over_reference: tuple) -> dict[str, dict]:
    """From the per-policy curves, the numbers of each retention figure, by the figure's file name: the prefill above
    the no-TTL floor at each TTL and the retained tokens over those at the reference TTL (prefill against storage);
    dead over active at a day under `ttl_only` (retained over active); and how far the reclaim-both policy lowers it
    (dead over active by policy)."""
    ttl_only = curves["ttl_only"]
    prefill_against_storage = {}
    for ttl in ttls_above_floor:
        prefill_against_storage[f"prefill_above_floor_{ttl}"] = float(ttl_only.normalised_prefill[ttl] - 1)
    for ttl in ttls_over_reference:
        retained = ttl_only.mean_retained_tokens[ttl] / ttl_only.mean_retained_tokens[REFERENCE_TTL]
        prefill_against_storage[f"retained_{ttl}_over_{REFERENCE_TTL}"] = float(retained)
    dead_at_a_day = ttl_only.dead_over_active["1d"]
    return {
        "prefill_against_storage": prefill_against_storage,
        "retained_over_active": {"dead_over_active_1d": float(dead_at_a_day)},
        "dead_over_active_by_policy": {
            "reclaim_both_dead_fall_1d": float(dead_at_a_day / curves["reclaim_both"].dead_over_active["1d"])},
    }
