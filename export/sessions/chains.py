"""Chains: each session's main chain, the turns on it, and whether it is an agent session.

A session is a root and every request that reaches it by following parents,
identified by the root's `request_id`. Its main chain runs from the root to
its deepest request; ties go to the latest `end_ms`, then the greatest
`request_id`. A request on the main chain is a `tool` turn when it was linked
by a tool id, or by text with a tool result following its last assistant
message; every other request, the root included, is a `user` turn. An agent
session has at least one tool turn and at least two user turns; SWE-bench
runs need only the tool turn.
"""

import collections
import dataclasses
import pathlib

import pyarrow
from pyarrow import parquet

from export import parallel, paths
from export.log_records import harness

MIN_TOOL_TURNS = 1  # an agent session has at least one tool turn
MIN_USER_TURNS = 2  # and at least two user turns; a benchmark run needs only the tool turn


CHAINS_SCHEMA = pyarrow.schema(
    [
        ("request_id", pyarrow.string()),
        ("session_id", pyarrow.string()),
        ("step", pyarrow.int32()),
        ("on_main_chain", pyarrow.bool_()),
        ("turn", pyarrow.string()),
        ("is_agent_session", pyarrow.bool_()),
    ]
)

TOOL_OUTPUT_KINDS = frozenset({"tool_result", "tool_output_as_text"})


@dataclasses.dataclass
class LinkedRequest:
    """One request with its link, and what the session step works out for it."""

    request_id: str
    parent_request_id: str | None
    link: str | None
    end_ms: int
    harness: str
    new_content_kind: str
    session_id: str = ""
    step: int = 0


def assign_sessions(requests: list[LinkedRequest]) -> None:
    """Give each request its session and step; `requests` are in order of `end_ms` then `request_id`,
    so every parent comes before its children."""
    by_id = {}
    for request in requests:
        if request.parent_request_id is None:
            request.session_id = request.request_id
            request.step = 0
        else:
            parent = by_id[request.parent_request_id]
            request.session_id = parent.session_id
            request.step = parent.step + 1
        by_id[request.request_id] = request


def main_chain(session_requests: list[LinkedRequest]) -> list[LinkedRequest]:
    """The path from the root to the deepest request, root first."""
    by_id = {request.request_id: request for request in session_requests}
    deepest = max(session_requests, key=lambda request: (request.step, request.end_ms, request.request_id))
    chain = [deepest]
    while chain[-1].parent_request_id is not None:
        chain.append(by_id[chain[-1].parent_request_id])
    return list(reversed(chain))


def turn_of(request: LinkedRequest) -> str:
    """`tool` when the request brings tool output back to the model, else `user`."""
    if request.link == "tool_id":
        return "tool"
    if request.link == "text" and request.new_content_kind in TOOL_OUTPUT_KINDS:
        return "tool"
    return "user"


def is_agent_session(harness_name: str, turns: list[str]) -> bool:
    """At least one tool turn and two user turns; a SWE-bench run needs only the tool turn."""
    tool_turns = turns.count("tool")
    user_turns = turns.count("user")
    if harness_name == harness.BENCHMARK_HARNESS:
        return tool_turns >= MIN_TOOL_TURNS
    return tool_turns >= MIN_TOOL_TURNS and user_turns >= MIN_USER_TURNS


def chain_rows_of_session(session_requests: list[LinkedRequest]) -> list[dict]:
    """The chains rows of one session: every request, with turns on the main chain only."""
    chain = main_chain(session_requests)
    turn_by_id = {request.request_id: turn_of(request) for request in chain}
    agent_session = is_agent_session(chain[0].harness, list(turn_by_id.values()))
    rows = []
    for request in session_requests:
        rows.append(
            {
                "request_id": request.request_id,
                "session_id": request.session_id,
                "step": request.step,
                "on_main_chain": request.request_id in turn_by_id,
                "turn": turn_by_id.get(request.request_id),
                "is_agent_session": agent_session,
            }
        )
    return rows


def chain_rows(requests: list[LinkedRequest]) -> list[dict]:
    """The chains rows of a week, session by session, each in request order."""
    ordered = sorted(requests, key=lambda request: (request.end_ms, request.request_id))
    assign_sessions(ordered)
    by_session = collections.defaultdict(list)
    for request in ordered:
        by_session[request.session_id].append(request)
    rows = []
    for session_id in sorted(by_session):
        rows.extend(chain_rows_of_session(by_session[session_id]))
    return rows


def read_linked_requests(week_directory: pathlib.Path) -> list[LinkedRequest]:
    """The week's requests with their links, from `signals.parquet` and `links.parquet`."""
    signal_columns = ["request_id", "end_ms", "harness", "new_content_kind"]
    signal_rows = parquet.read_table(week_directory / paths.SIGNALS_FILE, columns=signal_columns).to_pylist()
    links = parquet.read_table(week_directory / paths.LINKS_FILE).to_pylist()
    link_by_id = {link["request_id"]: link for link in links}
    requests = []
    for row in signal_rows:
        link = link_by_id[row["request_id"]]
        requests.append(
            LinkedRequest(
                request_id=row["request_id"],
                parent_request_id=link["parent_request_id"],
                link=link["link"],
                end_ms=row["end_ms"],
                harness=row["harness"],
                new_content_kind=row["new_content_kind"],
            )
        )
    return requests


def write_week_chains(week: paths.Week) -> dict:
    """Write the week's `chains.parquet`, and return its session counts."""
    rows = chain_rows(read_linked_requests(week.folder))
    parallel.write_rows(rows, CHAINS_SCHEMA, week.folder / paths.CHAINS_FILE)
    agent_sessions = {row["session_id"] for row in rows if row["is_agent_session"]}
    return {"sessions": len({row["session_id"] for row in rows}), "agent_sessions": len(agent_sessions)}
