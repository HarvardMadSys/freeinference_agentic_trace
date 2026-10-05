"""The release-sessions step: the week's release sessions, and their requests, spawn calls and counts.

The release sessions are the week's agent sessions and every session they spawned,
directly or not. A top-level session is a release session that no
release session spawned; each release session belongs to the trace of
the top-level session it descends from, and `trace_id` is that session's id.

The step writes, for the release sessions:
- `requests.parquet`: every main-chain request, with its session, step, turn
  and link, and its small columns, in the weekly log's order. It holds no
  bodies and no harness: the bodies stay in `extract.parquet`, and the
  harness is the session's.
- `sessions.parquet`: one row per release session.
- `spawns.parquet`: every spawn call on the main chain of a release session.
- `counts.json`: the week's counts, from its lines to its matched spawn calls.

The audit trail of every kept request, branches included, stays in
`links.parquet` and `chains.parquet`.
"""

import dataclasses
import json
import pathlib

import pyarrow
from pyarrow import compute, parquet

from export import parallel, paths
from export.sessions import extract, spawns

# The columns copied from the extract, in this order, then the four a session gives each request.
REQUEST_COLUMNS = [
    "request_id", "user_id", "model_id", "provider", "stream", "end_ms", "latency_ms", "ttft_ms",
    "prompt_tokens", "completion_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens",
]
SESSION_FIELDS = [
    ("session_id", pyarrow.string()),
    ("step", pyarrow.int32()),
    ("turn", pyarrow.string()),
    ("link", pyarrow.string()),
]


@dataclasses.dataclass(frozen=True)
class ReleaseRequest:
    """One release request as the later steps read it: where it sits, who supplied its input, and what follows it."""

    request_id: str
    session_id: str
    step: int
    turn: str  # `user` or `tool`: who supplied its new input
    harness: str  # its session's
    next_request_id: str | None  # the session's next request, which carries this one's results; None for the last


def read_requests(week_directory: pathlib.Path) -> list[ReleaseRequest]:
    """The week's release requests, in session and step order."""
    requests = parquet.read_table(week_directory / paths.REQUESTS_FILE,
                                  columns=["request_id", "session_id", "step", "turn"]).to_pylist()
    sessions = parquet.read_table(week_directory / paths.SESSIONS_FILE, columns=["session_id", "harness"]).to_pylist()
    harness_of_session = {session["session_id"]: session["harness"] for session in sessions}
    requests.sort(key=lambda request: (request["session_id"], request["step"]))
    population = []
    for position, request in enumerate(requests):
        following = requests[position + 1] if position + 1 < len(requests) else None
        same_session = following is not None and following["session_id"] == request["session_id"]
        population.append(ReleaseRequest(
            request_id=request["request_id"], session_id=request["session_id"], step=request["step"], turn=request["turn"],
            harness=harness_of_session[request["session_id"]], next_request_id=following["request_id"] if same_session else None,
        ))
    return population


REQUESTS_SCHEMA = pyarrow.schema(
    [extract.EXTRACT_SCHEMA.field(name) for name in REQUEST_COLUMNS]
    + [pyarrow.field(name, arrow_type) for name, arrow_type in SESSION_FIELDS]
)

SESSIONS_SCHEMA = pyarrow.schema(
    [
        ("session_id", pyarrow.string()),
        ("trace_id", pyarrow.string()),
        ("user_id", pyarrow.string()),
        ("harness", pyarrow.string()),
        ("is_agent_session", pyarrow.bool_()),
        ("n_requests", pyarrow.int32()),
        ("n_user_turns", pyarrow.int32()),
        ("n_tool_turns", pyarrow.int32()),
        ("start_ms", pyarrow.int64()),
        ("end_ms", pyarrow.int64()),
        ("spawned_by_session_id", pyarrow.string()),
        ("spawned_by_request_id", pyarrow.string()),
        ("spawned_by_call_index", pyarrow.int32()),
    ]
)



def spawner_by_child(matches: list[dict]) -> dict[str, dict]:
    """Each matched child session's spawn call; a session is the child of at most one call."""
    spawners = {}
    for match in matches:
        if match["child_session_id"] is not None:
            spawners[match["child_session_id"]] = match
    return spawners


def sessions_in_release(agent_sessions: set[str], matches: list[dict]) -> set[str]:
    """The agent sessions, and every session they spawned, recursively."""
    children_by_session = {}
    for match in matches:
        if match["child_session_id"] is not None:
            children_by_session.setdefault(match["session_id"], []).append(match["child_session_id"])
    population = set(agent_sessions)
    to_visit = sorted(agent_sessions)
    while to_visit:
        session_id = to_visit.pop()
        for child in children_by_session.get(session_id, []):
            if child not in population:
                population.add(child)
                to_visit.append(child)
    return population


def spawner_in_release(session_id: str, spawners: dict[str, dict], population: set[str]) -> dict | None:
    """The spawn call that started this session, when a release session made it."""
    spawner = spawners.get(session_id)
    if spawner is None or spawner["session_id"] not in population:
        return None
    return spawner


def trace_id(session_id: str, spawners: dict[str, dict], population: set[str]) -> str:
    """The id of the top-level session this release session descends from."""
    seen = {session_id}
    spawner = spawner_in_release(session_id, spawners, population)
    while spawner is not None and spawner["session_id"] not in seen:
        session_id = spawner["session_id"]
        seen.add(session_id)
        spawner = spawner_in_release(session_id, spawners, population)
    return session_id



def week_session_rows(chains: list[dict], signal_by_id: dict, members: set[str], spawners: dict) -> list[dict]:
    """One row per release session, from its main chain."""
    main_chains = {}
    for chain in chains:
        if chain["on_main_chain"] and chain["session_id"] in members:
            main_chains.setdefault(chain["session_id"], []).append(chain)
    rows = []
    for session_id in sorted(main_chains):
        chain = main_chains[session_id]
        root = signal_by_id[session_id]
        spawner = spawner_in_release(session_id, spawners, members)
        turns = [request["turn"] for request in chain]
        rows.append(
            {
                "session_id": session_id,
                "trace_id": trace_id(session_id, spawners, members),
                "user_id": root["user_id"],
                "harness": root["harness"],
                "is_agent_session": chain[0]["is_agent_session"],
                "n_requests": len(chain),
                "n_user_turns": turns.count("user"),
                "n_tool_turns": turns.count("tool"),
                "start_ms": root["start_ms"],
                "end_ms": max(signal_by_id[request["request_id"]]["end_ms"] for request in chain),
                "spawned_by_session_id": spawner["session_id"] if spawner else None,
                "spawned_by_request_id": spawner["request_id"] if spawner else None,
                "spawned_by_call_index": spawner["call_index"] if spawner else None,
            }
        )
    return rows


def write_requests(extract_file: pathlib.Path, chain_by_id: dict, output: pathlib.Path) -> int:
    """Copy the population's main-chain requests from the extract, adding session, step, turn and link."""
    kept_ids = pyarrow.array(sorted(chain_by_id), type=pyarrow.string())
    extract_parquet = parquet.ParquetFile(extract_file)
    written = 0
    with parallel.completed_file(output) as partial, parquet.ParquetWriter(partial, REQUESTS_SCHEMA, compression="zstd") as writer:
        for row_group in range(extract_parquet.num_row_groups):
            table = extract_parquet.read_row_group(row_group, columns=REQUEST_COLUMNS)
            table = table.filter(compute.is_in(table["request_id"], value_set=kept_ids))
            if table.num_rows == 0:
                continue
            request_ids = table["request_id"].to_pylist()
            for name, arrow_type in SESSION_FIELDS:
                values = [chain_by_id[request_id][name] for request_id in request_ids]
                table = table.append_column(name, pyarrow.array(values, type=arrow_type))
            writer.write_table(table)
            written += table.num_rows
    return written


def week_counts(week_directory: pathlib.Path, links: list[dict], chains: list[dict], members: set, spawn_rows: list[dict]) -> dict:
    """The week's `counts.json`: its extract counts, then links, sessions and spawn calls."""
    counts = json.loads((week_directory / paths.EXTRACT_COUNTS_FILE).read_text())
    agent_sessions = {chain["session_id"] for chain in chains if chain["is_agent_session"]}
    counts["requests_linked"] = sum(1 for link in links if link["parent_request_id"] is not None)
    counts["sessions"] = len({chain["session_id"] for chain in chains})
    counts["agent_sessions"] = len(agent_sessions)
    counts["population_sessions"] = len(members)
    counts["spawn_calls"] = len(spawn_rows)
    counts["spawn_calls_matched"] = sum(1 for row in spawn_rows if row["child_session_id"] is not None)
    agent_calls = [row for row in spawn_rows if row["session_id"] in agent_sessions]
    counts["agent_spawn_calls"] = len(agent_calls)
    counts["agent_spawn_calls_matched"] = sum(1 for row in agent_calls if row["child_session_id"] is not None)
    return counts


def write_week_release_sessions(week: paths.Week) -> dict:
    """Write the week's population outputs, and return its counts."""
    chains = parquet.read_table(week.folder / paths.CHAINS_FILE).to_pylist()
    links = parquet.read_table(week.folder / paths.LINKS_FILE).to_pylist()
    matches = parquet.read_table(week.folder / paths.SPAWN_MATCHES_FILE).to_pylist()
    signal_columns = ["request_id", "user_id", "harness", "start_ms", "end_ms"]
    signal_rows = parquet.read_table(week.folder / paths.SIGNALS_FILE, columns=signal_columns).to_pylist()
    signal_by_id = {row["request_id"]: row for row in signal_rows}
    link_by_id = {link["request_id"]: link["link"] for link in links}
    agent_sessions = {chain["session_id"] for chain in chains if chain["is_agent_session"]}
    members = sessions_in_release(agent_sessions, matches)
    spawners = spawner_by_child(matches)
    chain_by_id = {}
    for chain in chains:
        if chain["on_main_chain"] and chain["session_id"] in members:
            chain_by_id[chain["request_id"]] = dict(chain, link=link_by_id[chain["request_id"]])
    spawn_rows = [match for match in matches if match["session_id"] in members]
    sessions = week_session_rows(chains, signal_by_id, members, spawners)
    parallel.write_rows(sessions, SESSIONS_SCHEMA, week.folder / paths.SESSIONS_FILE)
    parallel.write_rows(spawn_rows, spawns.SPAWNS_SCHEMA, week.folder / paths.SPAWNS_FILE)
    write_requests(week.folder / paths.EXTRACT_FILE, chain_by_id, week.folder / paths.REQUESTS_FILE)
    counts = week_counts(week.folder, links, chains, members, spawn_rows)
    parallel.write_json(counts, week.folder / paths.COUNTS_FILE)
    return counts
