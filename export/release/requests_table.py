"""The release's `requests.parquet`: one row per released request, ordered by its hashed session id and step.

A row's facts come from the week's private files: the request's own row
(times, tokens, turn and link), its digest (its messages, their tokens per
role, its tool definitions' tokens and names, and its block hashes, released
as week-local ids, never as hashes), its transition from the step before, its
response, read from `extract.parquet` for the reason it ended, and the
categories of its calls (`tool_calls.parquet`), which decide what the next
request waited for. A declared tool's name ships as a call's does
(`public_words`): lower-cased when public, else `other`. No text is written.
"""

import collections
import dataclasses
import pathlib

import pandas
import pyarrow
from pyarrow import compute, parquet

from release import schema, trace_lines
from export import parallel, paths
from export.privacy import pseudonyms, public_words
from export.log_records import messages, responses

WEEK_REQUEST_COLUMNS = ["request_id", "user_id", "model_id", "provider", "stream", "end_ms", "latency_ms", "ttft_ms",
                     "session_id", "step", "turn", "link", "prompt_tokens", "completion_tokens", "cache_read_tokens"]
TOKEN_COLUMNS = ["prompt_tokens", "completion_tokens", "cache_read_tokens"]
BLOCK_IDS = "block_ids"
LINK_REASONS = {"tool_id": "tool_call_id", "text": "assistant_message"}  # the week's link, as the release names it
# The reasons the two response shapes state for ending, as the release names them; any other is `other`.
FINISH_REASONS = {
    "stop": "stop", "end_turn": "stop", "stop_sequence": "stop",
    "length": "length", "max_tokens": "length",
    "tool_calls": "tool_calls", "function_call": "tool_calls", "tool_use": "tool_calls",
    "content_filter": "content_filter", "refusal": "content_filter",
}
OTHER_FINISH_REASON = "other"
TOOL_WAITED_FOR = frozenset({"tool", "ask_user_tool", "subagent"})  # what a wait caused by a request's calls is


class UnknownRole(Exception):
    """An input message has a role the release does not count."""


@dataclasses.dataclass(frozen=True)
class InputFacts:
    """What the release keeps of a request's input, from its digest."""

    n_messages: int  # the messages, the tool definitions not counted
    role_tokens: dict[str, int]  # the o200k tokens of each role, and of the tool definitions
    tool_definition_names: list[str]  # the declared tools' names, as they ship


@dataclasses.dataclass(frozen=True)
class RequestFacts:
    """What a release row holds beyond the request's own row: its input, how its response ended, and its
    transition from the step before."""

    input: InputFacts
    finish_reason: str | None
    transition: str | None  # None for a session's first request
    cause: str | None  # the cause of a mutation; None otherwise


def finish_reason(stated: str | None) -> str | None:
    """The release's name for the reason a response stated; None when it stated none."""
    if stated is None:
        return None
    return FINISH_REASONS.get(stated, OTHER_FINISH_REASON)


def input_facts(digest: dict, words: public_words.PublicWords) -> InputFacts:
    """A request's input facts from its digest: its messages, the tokens of each role and of its tool definitions,
    and the tools' names."""
    role_tokens = {field: 0 for field in schema.ROLE_TOKEN_FIELDS}
    message_count = 0
    for role, tokens in zip(digest["message_roles"], digest["message_tokens"]):
        if role not in role_tokens:
            raise UnknownRole(f"request {digest['request_id']}: a message of role {role!r}")
        role_tokens[role] += tokens
        message_count += role != messages.TOOL_DEFINITIONS
    names = [public_words.released_name(name, words) for name in digest["tool_definition_names"]]
    return InputFacts(n_messages=message_count, role_tokens=role_tokens, tool_definition_names=names)


def finish_reasons_of_row_groups(extract_file: pathlib.Path, first: int, stop: int, wanted: pyarrow.Array) -> dict[str, str | None]:
    """The finish reason of each wanted request in row groups `first` up to `stop` of the extract, by request id."""
    extract = parquet.ParquetFile(extract_file)
    reasons = {}
    for row_group in range(first, stop):
        table = extract.read_row_group(row_group, columns=["request_id", "response"])
        table = table.filter(compute.is_in(table["request_id"], value_set=wanted))
        for request in table.to_pylist():
            reasons[request["request_id"]] = finish_reason(responses.read_response(request["response"]).stated_reason)
    return reasons


def finish_reasons(extract_file: pathlib.Path, request_ids: list[str], processes: int) -> dict[str, str | None]:
    """The finish reason of each given request, read from the extract's responses in parallel."""
    wanted = pyarrow.array(sorted(request_ids), type=pyarrow.string())
    reasons = {}
    with parallel.process_pool(processes) as pool:
        jobs = [pool.submit(finish_reasons_of_row_groups, extract_file, first, stop, wanted)
                for first, stop in parallel.row_group_ranges(extract_file, processes)]
        for job in jobs:
            reasons.update(job.result())
    return reasons


def read_request_facts(week: paths.Week, request_ids: list[str], words: public_words.PublicWords) -> dict[str, RequestFacts]:
    """The facts of each given request of the week, by raw request id, from its digest, its response and its
    transition."""
    digest_columns = ["request_id", "message_roles", "message_tokens", "tool_definition_names"]
    digests = parquet.read_table(week.folder / paths.DIGESTS_FILE, columns=digest_columns).to_pylist()
    inputs = {digest["request_id"]: input_facts(digest, words) for digest in digests}
    reasons = finish_reasons(week.folder / paths.EXTRACT_FILE, request_ids, week.processes)
    transitions = parquet.read_table(week.folder / paths.TRANSITIONS_FILE, columns=["request_id", "transition", "cause"])
    transition_of = {row["request_id"]: row for row in transitions.to_pylist()}
    facts = {}
    for request_id in request_ids:
        transition = transition_of.get(request_id, {"transition": None, "cause": None})
        facts[request_id] = RequestFacts(input=inputs[request_id], finish_reason=reasons[request_id],
                                         transition=transition["transition"], cause=transition["cause"])
    return facts


def call_categories(tool_calls_file: pathlib.Path) -> dict[str, set[str]]:
    """The categories of each request's calls, as the private tool calls hold them, calls written as text included."""
    table = parquet.read_table(tool_calls_file, columns=["request_id", "category"])
    categories = collections.defaultdict(set)
    for request_id, category in zip(table["request_id"].to_pylist(), table["category"].to_pylist()):
        categories[request_id].add(category)
    return categories


def request_row(request: dict, facts: RequestFacts, waited_for: str | None, keys: pseudonyms.PrivateKeys) -> dict:
    """One release row of a week's request; it keeps its raw `request_id` until the block hashes are joined."""
    row = {
        "request_id": request["request_id"],
        "session_id": pseudonyms.session_hash(request["session_id"], keys.salt),
        "step": request["step"],
        "link_reason": LINK_REASONS[request["link"]] if request["step"] > 0 else None,
        "start_ms": request["end_ms"] - (request["latency_ms"] or 0),
        "end_ms": request["end_ms"],
        "ttft_ms": request["ttft_ms"],
        "model": request["model_id"],
        "provider": pseudonyms.letter_of(request["provider"], keys.provider_letters),
        "stream": request["stream"],
        "step_trigger": request["turn"],
        "waited_for": waited_for,
        "role_tokens": facts.input.role_tokens,
        "n_messages": facts.input.n_messages,
        "tool_definition_names": facts.input.tool_definition_names,
        "finish_reason": facts.finish_reason,
        "transition": facts.transition,
        "cause": facts.cause,
    }
    for column in TOKEN_COLUMNS:
        row[column] = request[column]
    return row


def request_rows(week_requests: list[dict], facts: dict[str, RequestFacts], categories_of: dict[str, set[str]],
                 keys: pseudonyms.PrivateKeys) -> list[dict]:
    """The release's request rows, ordered by hashed session id and step."""
    ordered = sorted(week_requests,
                     key=lambda request: (pseudonyms.session_hash(request["session_id"], keys.salt), request["step"]))
    rows = []
    for position, request in enumerate(ordered):
        # A request after its session's first follows the previous row, since the rows are in session and step order.
        previous = categories_of.get(ordered[position - 1]["request_id"], set()) if request["step"] > 0 else set()
        waited_for = trace_lines.waited_for(request["step"], request["turn"], previous)
        rows.append(request_row(request, facts[request["request_id"]], waited_for, keys))
    return rows


def tool_latencies(rows: list[dict]) -> dict[schema.RequestKey, int]:
    """Per request whose next request waited for its tool calls, by session and step: that wait, in ms."""
    latencies = {}
    for row, next_row in zip(rows, rows[1:]):
        if next_row["session_id"] == row["session_id"] and next_row["waited_for"] in TOOL_WAITED_FOR:
            latencies[schema.RequestKey(row["session_id"], row["step"])] = next_row["start_ms"] - row["end_ms"]
    return latencies


def block_hashes(digests_file: pathlib.Path, request_ids: list[str]) -> pyarrow.Array:
    """The block hashes of the given requests, in their order; a request without a digest stops the step."""
    digests = parquet.read_table(digests_file, columns=["request_id", "block_hashes"])
    positions = compute.index_in(pyarrow.array(request_ids, pyarrow.string()), value_set=digests["request_id"])
    if positions.null_count:
        raise KeyError(f"{positions.null_count} released requests have no digest")
    hashes = digests["block_hashes"].cast(pyarrow.large_list(pyarrow.uint64()))
    return hashes.take(positions).combine_chunks()


def week_block_ids(hashes: pyarrow.Array) -> pyarrow.Array:
    """The requests' block hashes as week-local ids: 1 for the first hash the week reads, 2 for the next new one,
    and so on, reading the requests in order; two ids are equal exactly when their hashes are."""
    first_read_order = pandas.factorize(hashes.values.to_numpy())[0] + 1
    return pyarrow.LargeListArray.from_arrays(hashes.offsets, pyarrow.array(first_read_order, pyarrow.int64()))


def requests_table(rows: list[dict], digests_file: pathlib.Path) -> pyarrow.Table:
    """`requests.parquet`'s table: the rows, each with its week-local block ids, without the raw request ids."""
    columns = [name for name in schema.REQUESTS_SCHEMA.names if name != BLOCK_IDS]
    table = pyarrow.Table.from_pylist([{name: row[name] for name in columns} for row in rows],
                                      schema=pyarrow.schema([schema.REQUESTS_SCHEMA.field(name) for name in columns]))
    hashes = block_hashes(digests_file, [row["request_id"] for row in rows])
    return table.append_column(schema.REQUESTS_SCHEMA.field(BLOCK_IDS), week_block_ids(hashes))
