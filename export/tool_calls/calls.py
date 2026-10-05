"""The tool-calls step: one private row per tool call of the week's release requests.

Each row holds the call as it was sent and what is decided from it, in full;
the release anonymises these columns with the public word list when it builds
a week (`privacy/tool_columns.py`). A call's result is the fresh result, in
the session's next request, that carries its id, matched as `results.py` says;
only the result's tokens and outcome are kept, never its text.

The bodies are read once from `extract.parquet`, one row-group shard per
process; the parts are joined, in order, with each call's result.

Cline and mini-swe-agent write their calls into the reply's text and their
results into the next user text. A response of theirs with no call, whose
next step is a tool turn, is read again for what those results name: Cline's
`[NAME …] Result:` names an element `<NAME>` (or `<use_NAME>`) of the reply,
and mini-swe-agent's `<returncode>` the reply's one fenced block. Each call
found so takes the result it was found from. Before the file is replaced,
`tool_turns.py` checks that every tool turn follows a call.
"""

import collections
import dataclasses
import shutil
import typing

import pyarrow
import tiktoken
from pyarrow import compute, parquet

from release import schema
from export import parallel, paths, configs
from export.log_records import harness, responses
from export.sessions import release_sessions
from export.tool_calls import arguments, categories, results, tool_turns

ROWS_PER_WRITE = 1_000  # call rows held in memory before a part file is written to
CLINE = "cline"  # writes calls as XML elements and results as `[NAME …] Result:` blocks
# The harnesses whose calls are found from their results: Cline's XML elements, and the benchmark harness's one
# fenced command, whose result follows `<returncode>`.
RESULT_LED_HARNESSES = frozenset({CLINE, harness.BENCHMARK_HARNESS})
RESULT_LED_PART_FILE = "found_from_results.parquet"  # the part file of the calls found from results, beside the row-group parts

CALL_COLUMNS = [
    ("request_id", pyarrow.string()),
    ("call_index", pyarrow.int32()),
    ("raw_name", pyarrow.string()),
    ("arguments", pyarrow.large_string()),
    ("call_id", pyarrow.string()),
    ("call_format", pyarrow.string()),
    ("canonical_name", pyarrow.string()),
    ("family", pyarrow.string()),
    ("category", pyarrow.string()),
    ("shell_purpose", pyarrow.string()),
    ("programs_full", pyarrow.list_(pyarrow.string())),
    ("joined_by", pyarrow.list_(pyarrow.string())),
    ("deciding_program_full", pyarrow.string()),
    ("parse_status", pyarrow.string()),
    ("arguments_skeleton_full", pyarrow.large_string()),
]
PART_SCHEMA = pyarrow.schema(CALL_COLUMNS)
TOOL_CALLS_SCHEMA = pyarrow.schema(
    CALL_COLUMNS[:1] + [("session_id", pyarrow.string()), ("step", pyarrow.int32())] + CALL_COLUMNS[1:]
    + [("result_tokens", pyarrow.int64()), ("outcome", pyarrow.string())]
)


@dataclasses.dataclass(frozen=True)
class PartFacts:
    """What the requests of a part left for the join: the results each request's input carries, and the visible
    text of each result-led request with no call, both by request id."""

    carried: dict[str, results.CarriedResults]
    replies: dict[str, str]


class FoundCall(typing.NamedTuple):
    """A call found in a reply's text, with the result that named it."""

    call: responses.Call
    result: results.TextResult


class CallPosition(typing.NamedTuple):
    """Where a call sits: its request, and its index among the request's calls."""

    request_id: str
    call_index: int


@dataclasses.dataclass(frozen=True)
class ResultLedCalls:
    """The calls found from their results: their rows, and each one's result by its position."""

    rows: list[dict]
    result_of: dict[CallPosition, results.TextResult]


def call_row(request_id: str, call_index: int, call: responses.Call, tool_tables: categories.ToolTables) -> dict:
    """One call's raw and full columns."""
    canonical = categories.canonical_name(call.name, tool_tables)
    call_family = categories.family(call.name, canonical, tool_tables)
    parsed = arguments.parsed_arguments(call.arguments_text)
    row = {"request_id": request_id, "call_index": call_index,
           "raw_name": call.name if isinstance(call.name, str) else None, "arguments": call.arguments_text,
           "call_id": None if call.id is None else str(call.id), "call_format": call.call_format,
           "canonical_name": canonical, "family": call_family, "shell_purpose": None, "programs_full": None,
           "joined_by": None, "deciding_program_full": None, "parse_status": None}
    facts = None
    if call_family == categories.SHELL_FAMILY:
        facts = arguments.shell_facts(parsed)
        deciding = categories.deciding_program(facts.programs, facts.joined_by, tool_tables)
        row.update(programs_full=facts.programs, joined_by=facts.joined_by,
                   deciding_program_full=None if deciding is None else facts.programs[deciding],
                   parse_status=facts.parse_status,
                   shell_purpose=categories.shell_purpose(facts.programs, deciding, facts.backgrounded, facts.skeleton,
                                                          tool_tables))
    row["category"] = categories.category(call_family, row["shell_purpose"], tool_tables)
    row["arguments_skeleton_full"] = arguments.full_skeleton(parsed, facts)
    return row


def read_row_groups(shard: parallel.Shard, wanted: pyarrow.Array, result_led: frozenset[str],
                    tables: configs.RuleTables) -> PartFacts:
    """Write the call rows of the wanted requests in the shard's row groups to its part file, and return what its
    requests left for the join: the results each carries, and the reply of each `result_led` request with no call."""
    encoding = tiktoken.get_encoding(schema.TOKENIZER)
    extract = parquet.ParquetFile(shard.file)
    carried, replies = {}, {}
    with parquet.ParquetWriter(shard.part_file, PART_SCHEMA, compression="none") as writer:
        rows = []
        for row_group in range(shard.first, shard.stop):
            table = extract.read_row_group(row_group, columns=["request_id", "prompt", "response"])
            table = table.filter(compute.is_in(table["request_id"], value_set=wanted))
            for request in table.to_pylist():
                request_id = request["request_id"]
                carried[request_id] = results.carried_results(request["prompt"], encoding, tables.outcome_rules)
                response = responses.read_response(request["response"])
                for call_index, call in enumerate(response.calls):
                    rows.append(call_row(request_id, call_index, call, tables.tool_tables))
                if request_id in result_led and not response.calls and response.visible_text:
                    replies[request_id] = response.visible_text
                if len(rows) >= ROWS_PER_WRITE:
                    writer.write_table(pyarrow.Table.from_pylist(rows, schema=PART_SCHEMA))
                    rows = []
        writer.write_table(pyarrow.Table.from_pylist(rows, schema=PART_SCHEMA))
    return PartFacts(carried=carried, replies=replies)


def read_week_parts(week: paths.Week, requests: list[release_sessions.ReleaseRequest], tables: configs.RuleTables,
                    shards: list[parallel.Shard]) -> PartFacts:
    """Write every shard's call rows in parallel, and return what every release request left for the join."""
    wanted = pyarrow.array(sorted(request.request_id for request in requests), type=pyarrow.string())
    result_led = frozenset(request.request_id for request in requests if request.harness in RESULT_LED_HARNESSES)
    facts = PartFacts(carried={}, replies={})
    with parallel.process_pool(week.processes) as pool:
        jobs = [pool.submit(read_row_groups, shard, wanted, result_led, tables) for shard in shards]
        for job in jobs:
            part = job.result()
            facts.carried.update(part.carried)
            facts.replies.update(part.replies)
    if len(facts.carried) != len(requests):
        raise ValueError(f"{week.name}: read {len(facts.carried)} of {len(requests)} release requests")
    return facts


def calls_from_results(harness_name: str, reply: str, found_in_text: list[results.TextResult]) -> list[FoundCall]:
    """The calls a reply wrote in its text, as its next step's text results name them, each with its result.

    Cline: the k-th result naming NAME takes the reply's k-th `<NAME>` element, or, when the reply has no `<NAME>`
    element, its k-th `<use_NAME>` element. mini-swe-agent: a `<returncode>` result takes the reply's one fenced
    block. A result that names nothing in the reply gives no call."""
    found = []
    if harness_name == CLINE:
        seen = collections.Counter()
        for result in found_in_text:
            if result.name is None:
                continue
            element = result.name if responses.named_element_call(reply, result.name, 0) else f"use_{result.name}"
            call = responses.named_element_call(reply, element, seen[element])
            seen[element] += 1
            if call is not None:
                found.append(FoundCall(call, result))
    if harness_name == harness.BENCHMARK_HARNESS:
        returncodes = [result for result in found_in_text if result.name is None]
        call = responses.fenced_command_call(reply)
        if returncodes and call is not None:
            found.append(FoundCall(call, returncodes[0]))
    return found


def result_led_rows(requests_by_id: dict[str, release_sessions.ReleaseRequest], facts: PartFacts,
                    tool_tables: categories.ToolTables) -> ResultLedCalls:
    """The rows of the calls found from their results, for the replies whose next step is a tool turn."""
    rows, result_of = [], {}
    for request_id, reply in sorted(facts.replies.items()):
        request = requests_by_id[request_id]
        next_request = requests_by_id.get(request.next_request_id)
        if next_request is None or next_request.turn != "tool":
            continue
        found = calls_from_results(request.harness, reply, facts.carried[next_request.request_id].as_text)
        for call_index, found_call in enumerate(found):
            rows.append(call_row(request_id, call_index, found_call.call, tool_tables))
            result_of[CallPosition(request_id, call_index)] = found_call.result
    return ResultLedCalls(rows=rows, result_of=result_of)


def results_of_part(part: pyarrow.Table, requests_by_id: dict[str, release_sessions.ReleaseRequest], facts: PartFacts,
                    result_led: ResultLedCalls) -> list:
    """Each call's result, in the part's row order: the result it was found from, else the fresh result in the
    session's next request, matched with its response's other calls; None when none answers it."""
    request_ids, call_indexes = part["request_id"].to_pylist(), part["call_index"].to_pylist()
    ids_of = collections.defaultdict(list)
    for request_id, call_id in zip(request_ids, part["call_id"].to_pylist()):
        ids_of[request_id].append(call_id)
    matched = {}
    for request_id, ids in ids_of.items():
        next_request_id = requests_by_id[request_id].next_request_id
        fresh = facts.carried[next_request_id].with_ids if next_request_id is not None else []
        matched[request_id] = results.match_results(ids, fresh)
    chosen = []
    position_in_response = collections.Counter()
    for request_id, call_index in zip(request_ids, call_indexes):
        found = result_led.result_of.get(CallPosition(request_id, call_index))
        chosen.append(found if found is not None else matched[request_id][position_in_response[request_id]])
        position_in_response[request_id] += 1
    return chosen


def complete_part(part: pyarrow.Table, requests_by_id: dict[str, release_sessions.ReleaseRequest], facts: PartFacts,
                  result_led: ResultLedCalls) -> pyarrow.Table:
    """A part's rows with their session and step, and their result."""
    request_ids = part["request_id"].to_pylist()
    columns = {name: part[name] for name in part.schema.names}
    columns["session_id"] = pyarrow.array([requests_by_id[request_id].session_id for request_id in request_ids], pyarrow.string())
    columns["step"] = pyarrow.array([requests_by_id[request_id].step for request_id in request_ids], pyarrow.int32())
    chosen = results_of_part(part, requests_by_id, facts, result_led)
    columns["result_tokens"] = pyarrow.array([None if result is None else result.tokens for result in chosen], pyarrow.int64())
    columns["outcome"] = pyarrow.array([None if result is None else result.outcome for result in chosen], pyarrow.string())
    return pyarrow.table({field.name: columns[field.name] for field in TOOL_CALLS_SCHEMA}, schema=TOOL_CALLS_SCHEMA)


def write_week_tool_calls(week: paths.Week, tables: configs.RuleTables) -> int:
    """Write the week's `tool_calls.parquet` for its release requests, and return how many calls it holds."""
    requests = release_sessions.read_requests(week.folder)
    requests_by_id = {request.request_id: request for request in requests}
    parts_directory = parallel.parts_directory(week.temporary_folder, "tool_call", week.name)
    shards = parallel.row_group_shards(week.folder / paths.EXTRACT_FILE, parts_directory, week.processes)
    facts = read_week_parts(week, requests, tables, shards)
    result_led = result_led_rows(requests_by_id, facts, tables.tool_tables)
    part_files = [shard.part_file for shard in shards] + [parts_directory / RESULT_LED_PART_FILE]
    parquet.write_table(pyarrow.Table.from_pylist(result_led.rows, schema=PART_SCHEMA), part_files[-1])
    written = 0
    with parallel.completed_file(week.folder / paths.TOOL_CALLS_FILE) as partial:
        with parquet.ParquetWriter(partial, TOOL_CALLS_SCHEMA, compression="zstd") as writer:
            for part_file in part_files:
                part = parquet.read_table(part_file)
                writer.write_table(complete_part(part, requests_by_id, facts, result_led))
                written += part.num_rows
        calls_written = parquet.read_table(partial, columns=["request_id", "session_id", "call_id"]).to_pylist()
        tool_turns.check_week_tool_turns(week, requests, calls_written, facts.carried)
    shutil.rmtree(parts_directory)
    return written
