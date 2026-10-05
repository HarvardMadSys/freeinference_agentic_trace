"""The population the paper studies: the top-level sessions of one scope, with their requests and tool calls.

The population is the top-level sessions (no `parent_session_id`), every one
of which is an agent session: a subagent's session is part of its parent's
trace, not a user's session of its own, even when it passes the agent-session
rule. A population's requests
carry their session's `scope`, `harness` and `user_id`, and `n_tool_calls`,
the number of their rows in the tool calls. A session's scope is `SWE-Bench` when its harness is
the SWE-bench scaffold, and `Agentic` otherwise; as in the paper, an analysis
uses the Agentic scope unless it says otherwise.
"""

from release import reader

BENCHMARK_HARNESS = "eval:swe-bench"
AGENTIC = "Agentic"
SWE_BENCH = "SWE-Bench"
ALL_SCOPES = "All"


def scope_of(harness: str) -> str:
    """`SWE-Bench` for the benchmark scaffold, `Agentic` for every other harness."""
    return SWE_BENCH if harness == BENCHMARK_HARNESS else AGENTIC


def sessions_in_scope(release: reader.Release, scope: str = AGENTIC) -> reader.Release:
    """The top-level agent sessions of a scope (or of `All`), with their requests in session and step order and
    their tool calls."""
    sessions = release.sessions[release.sessions.parent_session_id.isna()].copy()
    sessions["scope"] = sessions.harness.map(scope_of)
    if scope != ALL_SCOPES:
        sessions = sessions[sessions.scope == scope]
    requests = release.requests[release.requests.session_id.isin(sessions.session_id)]
    requests = requests.merge(sessions[["session_id", "scope", "harness", "user_id"]], on="session_id")
    tool_calls = release.tool_calls[release.tool_calls.session_id.isin(sessions.session_id)].reset_index(drop=True)
    calls_per_request = tool_calls.groupby(["session_id", "step"]).size().rename("n_tool_calls").reset_index()
    requests = requests.merge(calls_per_request, on=["session_id", "step"], how="left")
    requests["n_tool_calls"] = requests.n_tool_calls.fillna(0).astype(int)
    requests = requests.sort_values(["session_id", "step"], ignore_index=True)
    return reader.Release(requests=requests, sessions=sessions.reset_index(drop=True), tool_calls=tool_calls)
