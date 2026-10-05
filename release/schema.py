"""The anonymous release's files and columns, defined once.

A published release week holds `traces/<YYYY-MM-DD>.jsonl`, one JSON line per
trace (`trace_lines.py`), and `manifest.json`, which holds the format version. Converted from those lines, the
tables the analysis reads hold one file per kind of row: `sessions.parquet`,
`requests.parquet` and `tool_calls.parquet`, keyed by `session_id`, then
`step`, then `call_index`, beside `tables.json`. A change to a file or a
column bumps `FORMAT_VERSION`, and every week is built again.
"""

import typing

import pyarrow

FORMAT_VERSION = 1  # the release's format; the reader reads this version only

SESSIONS_FILE = "sessions.parquet"
REQUESTS_FILE = "requests.parquet"
TOOL_CALLS_FILE = "tool_calls.parquet"
MANIFEST_FILE = "manifest.json"
TRACES_FOLDER = "traces"  # the published lines, one file per UTC day
TABLES_FILE = "tables.json"  # what the converted tables came from
BLOCK_TOKENS = 16  # tokens per block hash
MILLISECONDS_PER_SECOND = 1_000  # the release's times are in milliseconds
TOKENIZER = "o200k_base"  # the tokenizer every token count of the release uses

STEP_TRIGGERS = ("user", "tool")  # who supplied a step's new input: a user message, or tool results
WAITED_FOR = ("user", "tool", "ask_user_tool", "subagent")
LINK_REASONS = ("tool_call_id", "assistant_message")
FINISH_REASONS = ("stop", "length", "tool_calls", "content_filter", "other")
ROLES = ("system", "user", "assistant", "tool")
TRANSITIONS = ("append", "mutation", "switch")
CAUSES = ("compaction", "tool_definitions", "system_prompt", "dropped_turns", "injection", "tool_result", "other")
# Every category a call is counted in: the native ones, the shell ones marked `*`, and Other.
TOOL_CATEGORIES = ("Read", "Edit", "Write", "Browser", "MCP", "Ask User", "Subagent", "*Interpreter", "*Search",
                   "*File Ops", "*Git/VCS", "*Network", "*Wait/Poll", "*Build/Test", "*Packages", "Other")
OTHER_CATEGORY = "Other"  # the category of a call no native family or shell purpose places
SUBAGENT_CATEGORY = "Subagent"  # the category of a call that starts a subagent; the next request waited for it
ASK_USER_CATEGORY = "Ask User"  # the category of a call that hands control to the person
PARSE_STATUSES = ("success", "partial", "failed")
OUTCOMES = ("ok", "error", "denied", "timeout")
OTHER_WORD = "other"  # a tool name too few accounts used

SESSION_ID_HEX_CHARACTERS = 32  # a hashed session id
USER_ID_HEX_CHARACTERS = 16  # a hashed account id


class RequestKey(typing.NamedTuple):
    """What identifies a request in the release: its session, and its step within it."""

    session_id: str
    step: int


class SpawnKey(typing.NamedTuple):
    """What identifies the tool call that spawned a subagent session: the parent's request, and the call's index."""

    session_id: str
    step: int
    call_index: int


# The parts of an input `role_tokens` counts: the four message roles, and the tool definitions, which are no message.
ROLE_TOKEN_FIELDS = ROLES + ("tool_definitions",)
ROLE_TOKENS_TYPE = pyarrow.struct([(field, pyarrow.int64()) for field in ROLE_TOKEN_FIELDS])

SESSIONS_SCHEMA = pyarrow.schema(
    [
        ("session_id", pyarrow.string()),
        ("user_id", pyarrow.string()),
        ("harness", pyarrow.string()),
        ("parent_session_id", pyarrow.string()),  # the session whose call spawned it; null for a top-level session
        ("spawned_by_step", pyarrow.int32()),  # the parent's step whose response spawned it
        ("spawned_by_call_index", pyarrow.int32()),  # the call of that response that spawned it
    ]
)

REQUESTS_SCHEMA = pyarrow.schema(
    [
        ("session_id", pyarrow.string()),
        ("step", pyarrow.int32()),
        ("link_reason", pyarrow.string()),  # why it continues step - 1; null at step 0
        ("start_ms", pyarrow.int64()),
        ("end_ms", pyarrow.int64()),
        ("ttft_ms", pyarrow.int64()),
        ("model", pyarrow.string()),
        ("provider", pyarrow.string()),  # the provider's anonymous letter
        ("stream", pyarrow.bool_()),
        ("step_trigger", pyarrow.string()),  # who supplied the step's new input: `user` or `tool`
        ("waited_for", pyarrow.string()),
        ("prompt_tokens", pyarrow.int64()),
        ("completion_tokens", pyarrow.int64()),
        ("cache_read_tokens", pyarrow.int64()),
        ("role_tokens", ROLE_TOKENS_TYPE),  # the input's o200k tokens per message role, and of its tool definitions
        ("n_messages", pyarrow.int32()),  # the messages of the input; the tool definitions are not one
        ("tool_definition_names", pyarrow.list_(pyarrow.string())),  # each declared name, lower-cased, or `other`
        ("finish_reason", pyarrow.string()),
        ("transition", pyarrow.string()),  # against step - 1; null at step 0
        ("cause", pyarrow.string()),  # null unless a mutation
        # the input's 16-token blocks as week-local ids, equal exactly when the blocks' chained hashes are equal;
        # a large list, since a week holds billions
        ("block_ids", pyarrow.large_list(pyarrow.int64())),
    ]
)

TOOL_CALLS_SCHEMA = pyarrow.schema(
    [
        ("session_id", pyarrow.string()),
        ("step", pyarrow.int32()),
        ("call_index", pyarrow.int32()),  # the call's position in its response, from 0
        ("name", pyarrow.string()),  # the tool name, lower-cased, or `other`; the category is kept either way
        ("category", pyarrow.string()),
        ("programs", pyarrow.list_(pyarrow.string())),  # shell calls only: each program label, or its stand-in
        ("parse_status", pyarrow.string()),  # shell calls only
        ("arguments_skeleton", pyarrow.string()),  # the arguments, every value replaced by its type
        ("outcome", pyarrow.string()),  # null when no result carried the call's id
        ("result_tokens", pyarrow.int64()),
        ("latency_ms", pyarrow.int64()),  # the wait after its request until the next; shared by a batch's calls
    ]
)

SCHEMAS = {
    SESSIONS_FILE: SESSIONS_SCHEMA,
    REQUESTS_FILE: REQUESTS_SCHEMA,
    TOOL_CALLS_FILE: TOOL_CALLS_SCHEMA,
}
