"""Context mutations: how often each transition happens, in how many sessions, and what prefill it causes.

A transition is a request after the first of its session, with the `transition`
and `cause` the release gives it. `switch` transitions (the model or provider changed) are left out:
they are routing, not context management.
"""

import pandas

from paper.metrics import gaps

# The mutation table's rows: the two transitions, then every cause of a mutation, in the paper's order, the tool
# definitions after the system prompt they sit next to.
TABLE_ROWS = (
    ("append", "Append"), ("mutation", "Mutation"), ("injection", "Injection"), ("system_prompt", "System prompt"),
    ("tool_definitions", "Tool definitions"), ("dropped_turns", "Dropped turns"), ("compaction", "Compaction"),
    ("tool_result", "Tool result drop"), ("other", "Other"),
)


def transition_rows(requests: pandas.DataFrame, misses: pandas.DataFrame) -> pandas.DataFrame:
    """The scope's transitions other than `switch`, each with its request's hit ratio and missed tokens."""
    columns = ["session_id", "step", "prompt_tokens", "cache_read_tokens", "transition", "cause"]
    frame = requests.loc[requests.transition.notna(), columns]
    frame = frame.merge(misses[["session_id", "step", "missed_tokens"]], on=["session_id", "step"], how="left")
    frame = frame[frame.transition != "switch"].copy()
    frame["hit_ratio"] = gaps.hit_ratio(frame)
    return frame


def is_row(frame: pandas.DataFrame, name: str) -> pandas.Series:
    """Whether each transition counts in the mutation table's row `name`: its transition or its cause."""
    if name in ("append", "mutation"):
        return frame.transition == name
    return frame.cause == name


def mutation_table(frame: pandas.DataFrame) -> pandas.DataFrame:
    """The mutation table: per row, the share of transitions, of sessions, and of the tokens missed with no TTL."""
    sessions = frame.session_id.nunique()
    sessions_with_mutation = set(frame.session_id[frame.transition == "mutation"])
    missed = frame.missed_tokens.sum()
    rows = []
    for name, label in TABLE_ROWS:
        chosen = is_row(frame, name)
        if name == "append":
            session_count = sessions - len(sessions_with_mutation)
        else:
            session_count = frame.session_id[chosen].nunique()
        rows.append({
            "row": name,
            "label": label,
            "steps_share": float(chosen.mean()),
            "sessions_share": session_count / sessions,
            "prefill_share": float(frame.missed_tokens[chosen].sum() / missed),
        })
    return pandas.DataFrame(rows)
