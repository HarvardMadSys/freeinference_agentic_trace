"""Tool calls: which categories agents call, how often each fails, how much result and wait each costs.

A request's tool wait is the gap before the session's next request, when that
request waited for a tool, a person's answer to a question, or a subagent. A
single-call request is a request that made exactly one call; its tool wait is
that call's. Calls in a batch share one wait, so waits are only counted for
single-call requests.
"""

import pandas

from release import schema
from paper.metrics import turns

FAILED_OUTCOMES = ("error", "denied", "timeout")
SHELL_CATEGORY_MARK = "*"
EXAMPLES_PER_CATEGORY = 8  # the example commands list at most this many programs per shell category
STAND_IN_MARK = "<"  # a released program label that holds `<` holds a stand-in for a hidden word
TOOL_WAITED_FOR = ("tool", "ask_user_tool", "subagent")  # what a wait caused by a request's calls is
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)  # the box plots' whiskers, box and median


def tool_waits(requests: pandas.DataFrame) -> pandas.Series:
    """Each request's tool wait in seconds: the gap before the session's next request, when that request waited
    for a tool, a person's answer to a question, or a subagent; missing otherwise.

    Requests must be in session and step order.
    """
    next_wait = turns.waits(requests).groupby(requests.session_id).shift(-1)
    next_waited_for = requests.waited_for.groupby(requests.session_id).shift(-1)
    return next_wait.where(next_waited_for.isin(TOOL_WAITED_FOR))


def request_facts(requests: pandas.DataFrame) -> pandas.DataFrame:
    """Per request: its key, its number of calls, its step trigger, its latency and its tool wait."""
    return pandas.DataFrame({
        "session_id": requests.session_id,
        "step": requests.step,
        "width": requests.n_tool_calls,
        "step_trigger": requests.step_trigger,
        "latency_s": (requests.end_ms - requests.start_ms) / schema.MILLISECONDS_PER_SECOND,
        "tool_wait_s": tool_waits(requests),
    })


def calls_frame(tool_calls: pandas.DataFrame, requests: pandas.DataFrame) -> pandas.DataFrame:
    """The calls of the given requests, each with its request's width, step trigger, latency and tool wait."""
    return tool_calls.merge(request_facts(requests), on=["session_id", "step"], how="inner")


def shell_share(calls: pandas.DataFrame) -> float:
    """The share of calls that are shell calls: calls whose command was read, so that their parse status is set.

    The parse status is set for every shell call and for no other call, and
    unlike the tool name it is never anonymised.
    """
    return float(calls.parse_status.notna().mean())


def batch_kinds(calls: pandas.DataFrame) -> pandas.Series:
    """Each call's batch: `single` when its request made one call, else `same` or `mixed` by its request's categories."""
    categories_per_request = calls.groupby(["session_id", "step"]).category.transform("nunique")
    kinds = pandas.Series("single", index=calls.index)
    kinds[(calls.width > 1) & (categories_per_request == 1)] = "same"
    kinds[(calls.width > 1) & (categories_per_request > 1)] = "mixed"
    return kinds


def share_by_category(values: pandas.Series, categories: pandas.Series) -> pandas.Series:
    """Each category's share of the sum of `values`."""
    return values.groupby(categories).sum() / values.sum()


def category_order(calls: pandas.DataFrame) -> list[str]:
    """The categories in the overview's order: the native ones by share of calls, then the shell ones by share, then
    Other."""
    shares = calls.category.value_counts()
    native = [category for category in schema.TOOL_CATEGORIES
              if not category.startswith(SHELL_CATEGORY_MARK) and category != schema.OTHER_CATEGORY]
    shell = [category for category in schema.TOOL_CATEGORIES if category.startswith(SHELL_CATEGORY_MARK)]
    by_share = sorted(native, key=lambda category: -shares.get(category, 0)) + \
        sorted(shell, key=lambda category: -shares.get(category, 0))
    return by_share + [schema.OTHER_CATEGORY]


def overview_table(calls: pandas.DataFrame, session_count: int) -> pandas.DataFrame:
    """The tool overview: per category, its shares of calls, sessions, result tokens and tool wait, its failure rate, and
    its share of calls in same-category and mixed batches, all in percent; then the row for every call."""
    single = calls[calls.width == 1]
    with_outcome = calls[calls.outcome.notna()]
    kinds = batch_kinds(calls)
    columns = {
        "calls": calls.category.value_counts(normalize=True),
        "sessions": calls.groupby("category").session_id.nunique() / session_count,
        "tokens": share_by_category(calls.result_tokens.fillna(0), calls.category),
        "time": share_by_category(single.tool_wait_s.fillna(0), single.category),
        "fail": with_outcome.outcome.isin(FAILED_OUTCOMES).groupby(with_outcome.category).mean(),
        "same": (kinds == "same").groupby(calls.category).mean(),
        "mixed": (kinds == "mixed").groupby(calls.category).mean(),
    }
    table = pandas.DataFrame(columns).reindex(category_order(calls)).fillna(0.0) * 100
    table.loc["All"] = {
        "calls": 100.0, "sessions": 100 * calls.session_id.nunique() / session_count, "tokens": 100.0, "time": 100.0,
        "fail": 100 * with_outcome.outcome.isin(FAILED_OUTCOMES).mean(),
        "same": 100 * (kinds == "same").mean(), "mixed": 100 * (kinds == "mixed").mean(),
    }
    return table


def example_commands(calls: pandas.DataFrame) -> dict[str, list[str]]:
    """The example commands: per shell category, the most frequent programs of its calls that ran exactly one program, ties in
    alphabetical order, leaving out every label that holds a stand-in (`<program>`, `mvn <word>`, `<packages>`)."""
    shell = calls[calls.category.str.startswith(SHELL_CATEGORY_MARK)]
    labels = [programs[0] if programs is not None and len(programs) == 1 else None for programs in shell.programs]
    shell = shell.assign(label=pandas.Series(labels, index=shell.index, dtype="object"))
    shell = shell[shell.label.notna() & ~shell.label.str.contains(STAND_IN_MARK, regex=False, na=False)]
    examples = {}
    for category in schema.TOOL_CATEGORIES:
        if not category.startswith(SHELL_CATEGORY_MARK):
            continue
        counts = shell.label[shell.category == category].value_counts()
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        examples[category] = [label for label, _ in ordered[:EXAMPLES_PER_CATEGORY]]
    return examples


def distribution(values: pandas.Series) -> dict:
    """The quantiles the box plots draw, the mean and the count of the finite values."""
    finite = values[values.notna()].astype(float)
    summary = {f"p{round(quantile * 100)}": float(finite.quantile(quantile)) for quantile in QUANTILES}
    summary["mean"] = float(finite.mean())
    summary["n"] = int(len(finite))
    return summary


def latency_by_category(calls: pandas.DataFrame) -> dict[str, dict]:
    """Per category, the distribution of the tool wait of its single-call requests, in seconds."""
    single = calls[(calls.width == 1) & calls.tool_wait_s.notna()]
    return {category: distribution(single.tool_wait_s[single.category == category])
            for category in schema.TOOL_CATEGORIES}


def result_tokens_by_category(calls: pandas.DataFrame) -> dict[str, dict]:
    """Per category, the distribution of the result tokens of its calls with a matched result."""
    matched = calls[calls.result_tokens.notna()]
    return {category: distribution(matched.result_tokens[matched.category == category])
            for category in schema.TOOL_CATEGORIES}


# ---- batching: how many calls a request makes at once, and what a batch saves over serial calls. A request's width
# is its number of calls; shares of widths are shares of calls, so a request of three calls counts three times.
WIDTH_BUCKETS = ("1", "2", "3", "4+")  # the batching profile's rows of calls per request
WIDEST_DRAWN = 8  # the latency-by-width figure draws requests of eight or more calls together
BAND_QUANTILES = (0.10, 0.25, 0.5, 0.75, 0.90)  # the latency-by-width figure's bands and median


def width_bucket(width: int) -> str:
    """A width's row in the batching profile."""
    return str(width) if width < len(WIDTH_BUCKETS) else WIDTH_BUCKETS[-1]


def batching_profile(calls: pandas.DataFrame, session_count: int) -> dict:
    """The batching profile in percent: sessions with a batch, calls by the width of their request, and the batch mix."""
    batched_sessions = calls.session_id[calls.width > 1].nunique()
    buckets = calls.width.map(width_bucket).value_counts(normalize=True)
    batches = calls[calls.width > 1].groupby(["session_id", "step"]).category.nunique()
    return {
        "sessions_batched": 100 * batched_sessions / session_count,
        "calls_per_request": {bucket: 100 * float(buckets.get(bucket, 0.0)) for bucket in WIDTH_BUCKETS},
        "same_tool": 100 * float((batches == 1).mean()),
        "mixed": 100 * float((batches > 1).mean()),
    }


def latency_by_width(calls: pandas.DataFrame) -> pandas.DataFrame:
    """Per width, the quantiles of the tool wait of requests without an ask-user call, and the serial line.

    The serial line is the width times the median tool wait plus the median
    latency of one-call requests: one more request, and one more tool, per call.
    """
    asks_user = (calls.category == schema.ASK_USER_CATEGORY).groupby([calls.session_id, calls.step]).transform("max")
    requests = calls[~asks_user & calls.tool_wait_s.notna()].drop_duplicates(["session_id", "step"])
    requests = requests.assign(drawn_width=requests.width.clip(upper=WIDEST_DRAWN))
    one_call = requests[requests.width == 1]
    step_s = float(one_call.tool_wait_s.median() + one_call.latency_s.median())
    rows = []
    for width, group in requests.groupby("drawn_width"):
        row = {"width": int(width), "requests": len(group), "serial_s": width * step_s}
        for quantile in BAND_QUANTILES:
            row[f"p{round(quantile * 100)}"] = float(group.tool_wait_s.quantile(quantile))
        rows.append(row)
    return pandas.DataFrame(rows)


def width_by_category(calls: pandas.DataFrame) -> pandas.Series:
    """Per category, the mean width of the requests whose first call is of it, leaving out Ask User.

    Each request counts once, so one runaway request of thousands of calls
    cannot outweigh every other request of its category.
    """
    first_calls = calls[(calls.call_index == 0) & (calls.category != schema.ASK_USER_CATEGORY)]
    return first_calls.groupby("category").width.mean().sort_values(ascending=False)
