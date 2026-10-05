"""A release week as JSON lines, one per trace, and back into its three tables.

A line is one top-level session, with the line's `week` and `day` in front of its
fields: its requests in step order, each request its tool calls, and each spawning
call the subagent session it started, nested the same way. The week's lines go to
one file per UTC day of the sessions' first start. `write_lines` and `read_tables`
are each other's inverse; for each kind of row, its writer and reader sit side by
side. A line holds facts only: `waited_for` computes a request's wait when the line
is read, and the release step calls the same function, so that the two agree.
"""

import dataclasses
import datetime
import hashlib
import json
import pathlib

import orjson
import pyarrow

from release import schema

UNPARSED = "<unparsed>"  # the skeleton of arguments that were not a JSON object; null in a line


class TraceError(Exception):
    """The tables do not form whole traces (a session written twice or not at all), or a line is not a session."""


@dataclasses.dataclass(frozen=True)
class WeekTables:
    """A week's three tables, each in its release order."""

    sessions: pyarrow.Table
    requests: pyarrow.Table
    tool_calls: pyarrow.Table


def day_of(start_ms: int) -> str:
    """The UTC day of a time in ms since the epoch, as YYYY-MM-DD."""
    moment = datetime.datetime.fromtimestamp(start_ms / schema.MILLISECONDS_PER_SECOND, datetime.timezone.utc)
    return moment.date().isoformat()


def skeleton_pairs(skeleton: str) -> list | None:
    """An arguments skeleton as its [key, value] pairs, in order, repeated keys kept; None for `<unparsed>`."""
    if skeleton == UNPARSED:
        return None
    return [list(pair) for pair in json.loads(skeleton, object_pairs_hook=list)]


def skeleton_text(pairs: list | None) -> str:
    """The skeleton's JSON object text from its pairs, exactly as the export writes it; `<unparsed>` for None."""
    if pairs is None:
        return UNPARSED
    members = [orjson.dumps(key).decode() + ": " + orjson.dumps(value).decode() for key, value in pairs]
    return "{" + ", ".join(members) + "}"


# ---- a tool call: its row in tool_calls.parquet, and its object in a line


def tool_call_line(row: dict, subagent: dict | None) -> dict:
    """A tool call as it appears in a line, with the session it spawned nested in it."""
    return {
        "call_index": row["call_index"], "name": row["name"], "category": row["category"],
        "programs": row["programs"], "parse_status": row["parse_status"],
        "arguments": skeleton_pairs(row["arguments_skeleton"]), "outcome": row["outcome"],
        "result_tokens": row["result_tokens"], "latency_ms": row["latency_ms"],
        "subagent": subagent,
    }


def tool_call_row(call: dict, key: schema.RequestKey) -> dict:
    """A tool call's row, from its object in a line and the request it belongs to."""
    return {
        "session_id": key.session_id, "step": key.step, "call_index": call["call_index"],
        "name": call["name"], "category": call["category"], "programs": call["programs"],
        "parse_status": call["parse_status"], "arguments_skeleton": skeleton_text(call["arguments"]),
        "outcome": call["outcome"],
        "result_tokens": call["result_tokens"], "latency_ms": call["latency_ms"],
    }


# ---- a request: its row in requests.parquet, and its object in a line


def request_line(row: dict, block_ids, turn_index: int, tool_calls: list[dict]) -> dict:
    """A request as it appears in a line, with its tool calls; `waited_for` is left out, since it follows."""
    mutation = {"transition": row["transition"], "cause": row["cause"]} if row["transition"] is not None else None
    return {
        "step": row["step"], "turn_index": turn_index, "step_trigger": row["step_trigger"],
        "link_reason": row["link_reason"],
        "start_ms": row["start_ms"], "end_ms": row["end_ms"], "ttft_ms": row["ttft_ms"],
        "model": row["model"], "provider": row["provider"], "stream": row["stream"],
        "provider_tokens": {"prompt": row["prompt_tokens"], "completion": row["completion_tokens"],
                            "cached": row["cache_read_tokens"]},
        "role_tokens": row["role_tokens"], "n_messages": row["n_messages"],
        "tool_definition_names": row["tool_definition_names"], "finish_reason": row["finish_reason"],
        "mutation": mutation,
        "block_ids": block_ids,
        "tool_calls": tool_calls,
    }


def request_row(request: dict, session_id: str, waited_for_value: str | None) -> dict:
    """A request's row without its block ids, from its object in a line and the wait computed for it."""
    tokens = request["provider_tokens"]
    mutation = request["mutation"] or {"transition": None, "cause": None}
    return {
        "session_id": session_id, "step": request["step"], "link_reason": request["link_reason"],
        "start_ms": request["start_ms"], "end_ms": request["end_ms"], "ttft_ms": request["ttft_ms"],
        "model": request["model"], "provider": request["provider"], "stream": request["stream"],
        "step_trigger": request["step_trigger"], "waited_for": waited_for_value,
        "prompt_tokens": tokens["prompt"], "completion_tokens": tokens["completion"],
        "cache_read_tokens": tokens["cached"], "role_tokens": request["role_tokens"],
        "n_messages": request["n_messages"], "finish_reason": request["finish_reason"],
        "tool_definition_names": request["tool_definition_names"],
        "transition": mutation["transition"], "cause": mutation["cause"],
    }


def waited_for(step: int, step_trigger: str, previous_categories: set[str]) -> str | None:
    """`user`, `subagent`, `ask_user_tool` or `tool`; None for a session's first request."""
    if step == 0:
        return None
    if step_trigger == "user":
        return "user"
    if schema.SUBAGENT_CATEGORY in previous_categories:
        return "subagent"
    if schema.ASK_USER_CATEGORY in previous_categories:
        return "ask_user_tool"
    return "tool"


# ---- a session: its row in sessions.parquet, and its object in a line


def session_line(row: dict, requests: list[dict]) -> dict:
    """A session as it appears in a line: its facts and its requests, in step order."""
    return {
        "session_id": row["session_id"], "user_id": row["user_id"], "harness": row["harness"],
        "start_ms": min(request["start_ms"] for request in requests),
        "end_ms": max(request["end_ms"] for request in requests),
        "requests": requests,
    }


def session_row(session: dict, spawner: schema.SpawnKey | None) -> dict:
    """A session's row, from its object in a line and the call that spawned it, if any."""
    return {
        "session_id": session["session_id"], "user_id": session["user_id"], "harness": session["harness"],
        "parent_session_id": spawner.session_id if spawner else None,
        "spawned_by_step": spawner.step if spawner else None,
        "spawned_by_call_index": spawner.call_index if spawner else None,
    }


@dataclasses.dataclass(frozen=True)
class WeekIndex:
    """The week's tables as rows, grouped for nesting: each session's requests, each request's calls, and the
    session each spawning call started."""

    sessions: dict[str, dict]  # session id -> its row
    requests_of: dict[str, list[tuple[dict, object]]]  # session id -> its (row, block ids) in step order
    calls_of: dict[schema.RequestKey, list[dict]]  # a request -> its call rows in call order
    child_of: dict[schema.SpawnKey, str]  # a spawning call -> the session it started


def week_index(tables: WeekTables) -> WeekIndex:
    """The week's tables grouped for nesting; the tables are in release order, so each group keeps it."""
    sessions = {row["session_id"]: row for row in tables.sessions.to_pylist()}
    requests_of = {}
    block_ids = tables.requests["block_ids"].combine_chunks()
    for position, row in enumerate(tables.requests.drop_columns(["block_ids"]).to_pylist()):
        requests_of.setdefault(row["session_id"], []).append((row, block_ids[position].values.to_numpy()))
    calls_of = {}
    for row in tables.tool_calls.to_pylist():
        calls_of.setdefault(schema.RequestKey(row["session_id"], row["step"]), []).append(row)
    child_of = {}
    for session_id, row in sessions.items():
        if row["parent_session_id"] is not None:
            spawner = schema.SpawnKey(row["parent_session_id"], row["spawned_by_step"], row["spawned_by_call_index"])
            child_of[spawner] = session_id
    return WeekIndex(sessions=sessions, requests_of=requests_of, calls_of=calls_of, child_of=child_of)


def nested_session(index: WeekIndex, session_id: str) -> dict:
    """A session's line object, its subagents nested under the calls that spawned them."""
    requests = []
    turn_index = 0
    for position, (row, block_ids) in enumerate(index.requests_of[session_id]):
        if position > 0 and row["step_trigger"] == "user":
            turn_index += 1
        calls = []
        for call in index.calls_of.get(schema.RequestKey(session_id, row["step"]), []):
            child = index.child_of.get(schema.SpawnKey(session_id, row["step"], call["call_index"]))
            calls.append(tool_call_line(call, nested_session(index, child) if child is not None else None))
        requests.append(request_line(row, block_ids, turn_index, calls))
    return session_line(index.sessions[session_id], requests)


def check_every_session_is_reached(index: WeekIndex) -> None:
    """Stop unless every released session is top-level or sits under a released call of a reached session."""
    reached = {session_id for session_id, row in index.sessions.items() if row["parent_session_id"] is None}
    to_visit = list(reached)
    while to_visit:
        session_id = to_visit.pop()
        for row, _ in index.requests_of.get(session_id, []):
            for call in index.calls_of.get(schema.RequestKey(session_id, row["step"]), []):
                child = index.child_of.get(schema.SpawnKey(session_id, row["step"], call["call_index"]))
                if child is not None and child not in reached:
                    reached.add(child)
                    to_visit.append(child)
    if len(reached) != len(index.sessions):
        raise TraceError(f"{len(index.sessions) - len(reached)} released sessions sit under no released call")


def write_day(path: pathlib.Path, sessions: list[dict], week_name: str, day: str) -> dict:
    """Write one day's traces, one line each; return the file's bytes, sha256 and lines."""
    checksum = hashlib.sha256()
    with open(path, "wb") as handle:
        for session in sessions:
            line = {"week": week_name, "day": day, **session}
            data = orjson.dumps(line, option=orjson.OPT_SERIALIZE_NUMPY) + b"\n"
            checksum.update(data)
            handle.write(data)
    return {"bytes": path.stat().st_size, "sha256": checksum.hexdigest(), "lines": len(sessions)}


def write_lines(tables: WeekTables, traces_directory: pathlib.Path, week_name: str) -> dict[str, dict]:
    """Write the week's traces into `traces_directory`, one file per UTC day of the sessions' first start; return each
    file's bytes, sha256 and lines, by its path under the week's folder."""
    index = week_index(tables)
    check_every_session_is_reached(index)
    top_level = [nested_session(index, session_id)
                 for session_id, row in index.sessions.items() if row["parent_session_id"] is None]
    top_level.sort(key=lambda session: (session["start_ms"], session["session_id"]))
    by_day = {}
    for session in top_level:
        by_day.setdefault(day_of(session["start_ms"]), []).append(session)
    traces_directory.mkdir(parents=True, exist_ok=True)
    return {f"{traces_directory.name}/{day}.jsonl": write_day(traces_directory / f"{day}.jsonl", sessions, week_name, day)
            for day, sessions in sorted(by_day.items())}


@dataclasses.dataclass
class WeekRows:
    """The rows a week's lines give, as they are read: each table's rows, and each request's block ids."""

    sessions: list = dataclasses.field(default_factory=list)
    requests: list = dataclasses.field(default_factory=list)
    block_ids: list = dataclasses.field(default_factory=list)  # one list per request, in the order of `requests`
    tool_calls: list = dataclasses.field(default_factory=list)


def add_session(rows: WeekRows, session: dict, spawner: schema.SpawnKey | None) -> None:
    """One session's rows, its requests', their calls', and those of the subagents nested in them."""
    rows.sessions.append(session_row(session, spawner))
    previous_categories = set()
    for request in session["requests"]:
        key = schema.RequestKey(session["session_id"], request["step"])
        wait = waited_for(key.step, request["step_trigger"], previous_categories)
        rows.requests.append(request_row(request, key.session_id, wait))
        rows.block_ids.append(request["block_ids"])
        previous_categories = {call["category"] for call in request["tool_calls"]}
        for call in request["tool_calls"]:
            rows.tool_calls.append(tool_call_row(call, key))
            if call["subagent"] is not None:
                add_session(rows, call["subagent"], schema.SpawnKey(key.session_id, key.step, call["call_index"]))


def requests_table(rows: WeekRows) -> pyarrow.Table:
    """The requests table: the rows in session and step order, each with its block ids."""
    order = sorted(range(len(rows.requests)), key=lambda position: (rows.requests[position]["session_id"],
                                                                     rows.requests[position]["step"]))
    table_schema = schema.REQUESTS_SCHEMA
    fields = [table_schema.field(name) for name in table_schema.names if name != "block_ids"]
    table = pyarrow.Table.from_pylist([rows.requests[position] for position in order], schema=pyarrow.schema(fields))
    block_ids = pyarrow.array([rows.block_ids[position] for position in order], table_schema.field("block_ids").type)
    return table.append_column(table_schema.field("block_ids"), block_ids)


def read_tables(day_files: list[pathlib.Path]) -> WeekTables:
    """A week's three tables, in their release order, from its day files."""
    rows = WeekRows()
    for path in day_files:
        with open(path, "rb") as handle:
            for number, line in enumerate(handle, start=1):
                session = orjson.loads(line)
                if "session_id" not in session:
                    raise TraceError(f"{path} line {number} is not a session; download the week's traces again")
                add_session(rows, session, None)
    sessions = sorted(rows.sessions, key=lambda row: row["session_id"])
    tool_calls = sorted(rows.tool_calls, key=lambda row: (row["session_id"], row["step"], row["call_index"]))
    return WeekTables(
        sessions=pyarrow.Table.from_pylist(sessions, schema=schema.SESSIONS_SCHEMA),
        requests=requests_table(rows),
        tool_calls=pyarrow.Table.from_pylist(tool_calls, schema=schema.TOOL_CALLS_SCHEMA),
    )
