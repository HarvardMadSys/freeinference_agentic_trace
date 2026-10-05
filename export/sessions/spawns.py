"""Spawn calls: an agent delegating work to a subagent, and the session that subagent ran.

A spawn call is a tool call whose name, lower-cased, is on the spawn list of
`export/configs/tool_categories.toml`. Its delegation prompt is repeated in the
child's first user message, so a call is matched to the root of another
session of the same account that starts from 30 s before to 10 minutes after
the spawning request completes and whose first user message contains the
prompt's opening. Calls are assigned in completion order; each takes the
nearest unclaimed match, so a session is the child of at most one call.

The spawns step reads `signals.parquet` and `chains.parquet` and writes
`spawn_matches.parquet`: every spawn call on a main chain, matched or not.
"""

import collections
import re

import orjson
import pyarrow
from pyarrow import parquet

from export import parallel, paths
from export.log_records import responses


BILLING_HEADER = re.compile(r"^\s*x-anthropic-billing-header:\s*(?:[\w.-]+=[^;\n]*;\s*)+")
WHITESPACE_RUN = re.compile(r"\s+")
# A spawn call's delegation prompt: the first of these arguments that is a string of 3 or more words,
PROMPT_ARGUMENTS = ("prompt", "task", "task_description", "message", "goal", "description")
MIN_PROMPT_WORDS = 3
TASK_LIST_ARGUMENT = "tasks"  # else a `tasks` string of more than 40 characters
MIN_TASK_LIST_CHARACTERS = 40
MAX_PROMPT_CHARACTERS = 20_000  # a prompt is cut to 20,000 characters before it is normalised
SPAWN_OPENING_CHARACTERS = 1_500  # the first 1,500 normalised characters must appear in the child
MAX_ROOT_TEXT_CHARACTERS = 30_000  # of the child root's first user message, cut to 30,000 characters
SPAWN_WINDOW_BEFORE_MS = 30_000  # a child may start up to 30 s before the spawning request completes
SPAWN_WINDOW_AFTER_MS = 600_000  # and up to 10 minutes after

SPAWNS_SCHEMA = pyarrow.schema(
    [
        ("session_id", pyarrow.string()),
        ("request_id", pyarrow.string()),
        ("call_index", pyarrow.int32()),
        ("tool_name", pyarrow.string()),
        ("has_prompt", pyarrow.bool_()),
        ("n_candidates", pyarrow.int32()),
        ("child_session_id", pyarrow.string()),
    ]
)


def normalise(text: str) -> str:
    """Without a leading billing header, whitespace runs as one space, stripped and lower-cased."""
    text = BILLING_HEADER.sub("", text, count=1)
    return WHITESPACE_RUN.sub(" ", text).strip().lower()


def delegation_prompt(arguments_text: str | None) -> str | None:
    """The prompt a spawn call hands its child: the first prompt-like argument of 3 or more words,
    else a long `tasks` argument; None when there is none."""
    if arguments_text is None:
        return None
    try:
        arguments = orjson.loads(arguments_text)
    except orjson.JSONDecodeError:
        return None
    if not isinstance(arguments, dict):
        return None
    for name in PROMPT_ARGUMENTS:
        value = arguments.get(name)
        if isinstance(value, str) and len(value.split()) >= MIN_PROMPT_WORDS:
            return value
    task_list = arguments.get(TASK_LIST_ARGUMENT)
    if isinstance(task_list, str) and len(task_list) > MIN_TASK_LIST_CHARACTERS:
        return task_list
    return None


def spawn_calls(response: responses.Response, spawn_tools: frozenset[str]) -> list[dict]:
    """The response's spawn calls: position among all its calls, raw name, and prompt opening or None."""
    calls = []
    for call_index, call in enumerate(response.calls):
        if not isinstance(call.name, str) or call.name.lower() not in spawn_tools:
            continue
        prompt = delegation_prompt(call.arguments_text)
        opening = None
        if prompt is not None:
            opening = normalise(prompt[: MAX_PROMPT_CHARACTERS])[: SPAWN_OPENING_CHARACTERS]
        calls.append({"call_index": call_index, "tool_name": call.name, "opening": opening})
    return calls


def first_user_text(input_messages: list) -> str:
    """The normalised text of the first user message, cut to 30,000 characters first."""
    for message in input_messages:
        if isinstance(message, dict) and message.get("role") == "user":
            return normalise(message_text(message)[: MAX_ROOT_TEXT_CHARACTERS])
    return ""


def message_text(message: dict) -> str:
    """A user message's own text: its string content, or its text blocks joined with spaces."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    texts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in (None, "text"):
            text = block.get("text")
            texts.append(text if isinstance(text, str) else "")
    return " ".join(texts)


def candidate_roots(spawner: dict, roots_by_user: dict) -> list[dict]:
    """The roots of the account's other sessions that start inside the spawn window."""
    if spawner["user_id"] is None:
        return []
    earliest = spawner["end_ms"] - SPAWN_WINDOW_BEFORE_MS
    latest = spawner["end_ms"] + SPAWN_WINDOW_AFTER_MS
    candidates = []
    for root in roots_by_user[spawner["user_id"]]:
        if root["session_id"] != spawner["session_id"] and earliest <= root["start_ms"] <= latest:
            candidates.append(root)
    return candidates


def match_spawns(spawners: list[dict], roots: list[dict]) -> list[dict]:
    """One row per spawn call of the main-chain `spawners`, with its matched child or None."""
    roots_by_user = collections.defaultdict(list)
    for root in roots:
        roots_by_user[root["user_id"]].append(root)
    claimed = set()
    rows = []
    for spawner in sorted(spawners, key=lambda request: (request["end_ms"], request["request_id"])):
        for call in spawner["spawn_calls"]:
            candidates = candidate_roots(spawner, roots_by_user)
            child = None
            if call["opening"]:
                child = nearest_unclaimed_match(call["opening"], spawner["end_ms"], candidates, claimed)
            if child is not None:
                claimed.add(child)
            rows.append(
                {
                    "session_id": spawner["session_id"],
                    "request_id": spawner["request_id"],
                    "call_index": call["call_index"],
                    "tool_name": call["tool_name"],
                    "has_prompt": call["opening"] is not None,
                    "n_candidates": len(candidates),
                    "child_session_id": child,
                }
            )
    return rows


def nearest_unclaimed_match(opening: str, end_ms: int, candidates: list[dict], claimed: set) -> str | None:
    """The unclaimed candidate whose first user text contains the opening, nearest in start time;
    ties go to the earlier start, then the smaller session id."""
    matches = [
        root for root in candidates if root["session_id"] not in claimed and opening in root["first_user_text"]
    ]
    if not matches:
        return None
    nearest = min(matches, key=lambda root: (abs(root["start_ms"] - end_ms), root["start_ms"], root["session_id"]))
    return nearest["session_id"]


def write_week_spawns(week: paths.Week) -> dict:
    """Write the week's `spawn_matches.parquet`, and return how many spawn calls there are and how many matched."""
    signal_columns = ["request_id", "user_id", "start_ms", "end_ms", "spawn_calls", "first_user_text"]
    signal_by_id = {}
    for row in parquet.read_table(week.folder / paths.SIGNALS_FILE, columns=signal_columns).to_pylist():
        signal_by_id[row["request_id"]] = row
    spawners = []
    roots = []
    for chain in parquet.read_table(week.folder / paths.CHAINS_FILE).to_pylist():
        request = dict(signal_by_id[chain["request_id"]], session_id=chain["session_id"])
        if chain["step"] == 0:
            roots.append(request)
        if chain["on_main_chain"] and request["spawn_calls"]:
            spawners.append(request)
    rows = match_spawns(spawners, roots)
    parallel.write_rows(rows, SPAWNS_SCHEMA, week.folder / paths.SPAWN_MATCHES_FILE)
    return {"spawn_calls": len(rows), "matched": sum(1 for row in rows if row["child_session_id"] is not None)}
