"""Token series from the providers' counts, their cost, and each role's share of an input.

Input, cached and output tokens are the providers' counts. A row is
cache-reporting when its provider reported cache reads; only such rows have an
uncached count. Role shares use the release's o200k counts per role.
"""

import pandas

from paper.metrics import prices

# The context-share lines and the `role_tokens` fields each one sums: the four message roles, as written.
MESSAGE_ROLES = {"System": ("system",), "User": ("user",), "Assistant": ("assistant",), "Tool": ("tool",)}
# The same, with the tool definitions counted beside the system prompt, as the fixed instructions of the input.
DEFINITIONS_AS_SYSTEM = {"System + Tools": ("system", "tool_definitions"), "User": ("user",),
                         "Assistant": ("assistant",), "Tool": ("tool",)}


def is_cache_reporting(requests: pandas.DataFrame) -> pandas.Series:
    """Whether the provider reported cache reads for the request."""
    return requests.cache_read_tokens.notna()


def uncached_tokens(requests: pandas.DataFrame) -> pandas.Series:
    """Input minus cached tokens, at least 0, on cache-reporting rows; missing otherwise."""
    uncached = (requests.prompt_tokens - requests.cache_read_tokens).clip(lower=0)
    return uncached.where(is_cache_reporting(requests))


def cost_components(requests: pandas.DataFrame) -> pandas.DataFrame:
    """Per cache-reporting request: the cost of its cached, uncached and output tokens."""
    rows = requests[is_cache_reporting(requests) & (requests.prompt_tokens > 0)]
    return pandas.DataFrame(
        {
            "session_id": rows.session_id,
            "cached": rows.cache_read_tokens * prices.CACHED_PRICE,
            "uncached": uncached_tokens(rows) * prices.UNCACHED_PRICE,
            "output": rows.completion_tokens.fillna(0) * prices.OUTPUT_PRICE,
        }
    )


def cost_shares_per_session(requests: pandas.DataFrame) -> pandas.DataFrame:
    """Per session with any cost: the shares of cached, uncached and output tokens in it."""
    costs = cost_components(requests).groupby("session_id")[["cached", "uncached", "output"]].sum()
    total = costs.sum(axis=1)
    costs = costs[total > 0]
    return costs.div(costs.sum(axis=1), axis=0)


def pooled_cost_shares(requests: pandas.DataFrame) -> dict[str, float]:
    """The cached, uncached and output shares of the whole cost."""
    totals = cost_components(requests)[["cached", "uncached", "output"]].sum()
    return (totals / totals.sum()).to_dict()


def role_shares(requests: pandas.DataFrame, parts: dict[str, tuple[str, ...]]) -> pandas.DataFrame:
    """Per request with any tokens in the parts: each part's share of their sum; a part sums its `role_tokens`
    fields."""
    sums = pandas.DataFrame({name: requests[[f"role_tokens.{field}" for field in fields]].sum(axis=1)
                             for name, fields in parts.items()})
    total = sums.sum(axis=1)
    return sums[total > 0].div(total[total > 0], axis=0)


def user_prompt_tokens(requests: pandas.DataFrame) -> pandas.Series:
    """Per user step, the tokens of its user's new prompt: its user tokens' rise over the step before, its whole
    user tokens at step 0; a step whose user tokens did not rise, as when earlier messages were rewritten, has none.

    Requests must be in session and step order.
    """
    user_tokens = requests["role_tokens.user"]
    rise = user_tokens - user_tokens.groupby(requests.session_id).shift()
    rise = rise.where(requests.step > 0, user_tokens)
    return rise[(requests.step_trigger == "user") & (rise > 0)]


def growth_rows(requests: pandas.DataFrame) -> pandas.DataFrame:
    """The requests token growth along the context is measured over: cache-reporting, with an input and at least
    one message."""
    usable = is_cache_reporting(requests) & (requests.prompt_tokens > 0) & (requests.n_messages >= 1)
    return requests[usable]
