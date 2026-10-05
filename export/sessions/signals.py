"""The linking signals of one request, computed from its own prompt and response.

- `emitted_ids`: keys of the tool-call ids in the response.
- `replayed_ids`: keys of the tool-call ids on the input's last assistant message.
- `answer_shingles`: the 8-word phrases of the response's visible answer.
- `replayed_shingles`: the 8-word phrases of the input's last assistant message.
- `new_content_kind`: what follows that message: a tool result, tool output
  written as text, a user message, or nothing.

Reasoning never enters a signal. Step 2, signals, reads a week's
`extract.parquet` and writes `signals.parquet`: one small row per kept request,
with what linking, sessions and spawn matching need and none of the bodies.
The extract's row groups are read by several processes, and the rows are
written in the extract's order.
"""

import hashlib
import pathlib

import pyarrow
from pyarrow import parquet

from export import parallel, paths
from export.log_records import call_ids, messages, responses
from export.sessions import spawns

TOOL_RESULT_BLOCK_TYPES = frozenset({"tool_result", "tool-result"})
SHINGLE_WORDS = 8  # a phrase is 8 consecutive words
MAX_SHINGLED_CHARACTERS = 8_000  # only the first 8,000 characters of each side are compared
MIN_VISIBLE_WORDS = 8  # a shorter answer has its tool calls appended before it is compared
MAX_NEW_CONTENT_CHARACTERS = 4_000  # characters of new content inspected for harness-written tool output


def shingles(text: str) -> set[int]:
    """The hashed 8-word phrases of the text; a text of 1 to 7 words is one phrase."""
    words = text.split()
    if not words:
        return set()
    if len(words) < SHINGLE_WORDS:
        return {phrase_hash(" ".join(words))}
    phrases = set()
    for start in range(len(words) - SHINGLE_WORDS + 1):
        phrases.add(phrase_hash(" ".join(words[start : start + SHINGLE_WORDS])))
    return phrases


def phrase_hash(phrase: str) -> int:
    """An unsigned 64-bit hash of a phrase: BLAKE2b with an 8-byte digest, read little-endian."""
    digest = hashlib.blake2b(phrase.encode("utf-8", errors="replace"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def emitted_ids(response: responses.Response) -> set[str]:
    """The keys of the ids of the response's tool calls."""
    keys = set()
    for call in response.calls:
        keys |= call_ids.emitted_keys(call.id)
    return keys


def answer_shingles(response: responses.Response) -> set[int]:
    """The phrases of the visible answer; a short answer has its calls appended first."""
    text = response.visible_text or ""
    if len(text.split()) < MIN_VISIBLE_WORDS and response.calls:
        rendered_calls = [f"{call.name or ''} {call.arguments_text or ''}" for call in response.calls]
        text = text + " " + " ".join(rendered_calls)
    return shingles(text[: MAX_SHINGLED_CHARACTERS])


def replayed_ids(input_messages: list) -> set[str]:
    """The keys of the tool-call ids on the last assistant message of the input."""
    position = messages.last_assistant_position(input_messages)
    if position is None:
        return set()
    last_assistant = input_messages[position]
    raw_ids = []
    tool_calls = last_assistant.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if isinstance(call, dict):
                raw_ids.append(call.get("id"))
    content = last_assistant.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                raw_ids.append(block.get("id"))
    keys = set()
    for raw_id in raw_ids:
        keys |= call_ids.replayed_keys(raw_id)
    return keys


def replayed_shingles(input_messages: list) -> set[int]:
    """The phrases of the last assistant message of the input."""
    position = messages.last_assistant_position(input_messages)
    if position is None:
        return set()
    return shingles(messages.plain_text(input_messages[position])[: MAX_SHINGLED_CHARACTERS])


def new_content_kind(input_messages: list) -> str:
    """What follows the last assistant message: `tool_result`, `tool_output_as_text`, `user_message` or `nothing`."""
    new = messages.new_messages(input_messages)
    if not new:
        return "nothing"
    for message in new:
        if message.get("role") == "tool" or has_tool_result_block(message):
            return "tool_result"
    texts = [messages.plain_text(message) for message in new]
    joined = "\n".join(text for text in texts if text)[: MAX_NEW_CONTENT_CHARACTERS]
    if any(pattern.search(joined) for pattern in messages.TOOL_OUTPUT_AS_TEXT):
        return "tool_output_as_text"
    return "user_message"


def has_tool_result_block(message: dict) -> bool:
    """Whether the message's content holds a tool-result block."""
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(block, dict) and block.get("type") in TOOL_RESULT_BLOCK_TYPES for block in content)


SIGNALS_SCHEMA = pyarrow.schema(
    [
        ("request_id", pyarrow.string()),
        ("user_id", pyarrow.string()),
        ("harness", pyarrow.string()),
        ("start_ms", pyarrow.int64()),
        ("end_ms", pyarrow.int64()),
        ("emitted_ids", pyarrow.list_(pyarrow.string())),
        ("replayed_ids", pyarrow.list_(pyarrow.string())),
        ("answer_shingles", pyarrow.list_(pyarrow.uint64())),
        ("replayed_shingles", pyarrow.list_(pyarrow.uint64())),
        ("new_content_kind", pyarrow.string()),
        (
            "spawn_calls",
            pyarrow.list_(
                pyarrow.struct(
                    [("call_index", pyarrow.int32()), ("tool_name", pyarrow.string()), ("opening", pyarrow.string())]
                )
            ),
        ),
        ("first_user_text", pyarrow.large_string()),
    ]
)

EXTRACT_COLUMNS = ["request_id", "user_id", "harness", "start_ms", "end_ms", "prompt", "response"]


def request_signals(request: dict, spawn_tools: frozenset[str]) -> dict:
    """The signals row of one kept request; sets are written sorted so that re-runs are identical."""
    input_messages = messages.messages_list(request["prompt"])
    response = responses.read_response(request["response"])
    return {
        "request_id": request["request_id"],
        "user_id": request["user_id"],
        "harness": request["harness"],
        "start_ms": request["start_ms"],
        "end_ms": request["end_ms"],
        "emitted_ids": sorted(emitted_ids(response)),
        "replayed_ids": sorted(replayed_ids(input_messages)),
        "answer_shingles": sorted(answer_shingles(response)),
        "replayed_shingles": sorted(replayed_shingles(input_messages)),
        "new_content_kind": new_content_kind(input_messages),
        "spawn_calls": spawns.spawn_calls(response, spawn_tools),
        "first_user_text": spawns.first_user_text(input_messages),
    }


def signals_of_row_groups(extract_file: pathlib.Path, first: int, stop: int, spawn_tools: frozenset[str]) -> list[dict]:
    """The signals rows of row groups `first` up to `stop` of the extract."""
    extract = parquet.ParquetFile(extract_file)
    rows = []
    for row_group in range(first, stop):
        for request in extract.read_row_group(row_group, columns=EXTRACT_COLUMNS).to_pylist():
            rows.append(request_signals(request, spawn_tools))
    return rows


def write_week_signals(week: paths.Week, spawn_tools: frozenset[str]) -> int:
    """Write the week's `signals.parquet` from its `extract.parquet`, and return its row count."""
    extract_file = week.folder / paths.EXTRACT_FILE
    written = 0
    with parallel.process_pool(week.processes) as pool, parallel.completed_file(week.folder / paths.SIGNALS_FILE) as partial:
        jobs = [pool.submit(signals_of_row_groups, extract_file, first, stop, spawn_tools)
                for first, stop in parallel.row_group_ranges(extract_file, week.processes)]
        with parquet.ParquetWriter(partial, SIGNALS_SCHEMA, compression="zstd") as writer:
            for job in jobs:
                rows = job.result()
                written += len(rows)
                if rows:
                    writer.write_table(pyarrow.Table.from_pylist(rows, schema=SIGNALS_SCHEMA))
    return written
