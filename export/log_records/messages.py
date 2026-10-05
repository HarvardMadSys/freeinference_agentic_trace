"""A request's input as it sent it, and the plain text of one message.

A record's `prompt` holds the messages, and its `tools` field the tool
definitions. A harness that sends its system prompt in the payload's top-level
`system` field (Claude Code) has it kept in `system_prompts.parquet`. The input
is, in this order, the order Anthropic documents for its prompt cache:
1. the tool definitions, when the request declared any: one entry of role
   `tool_definitions`, the `tools` list with every `cache_control` and
   `signature` key removed at any depth, since they mark the cache and carry
   no text the model reads;
2. the top-level system prompt, when there is one, as a message with role
   `system`;
3. the prompt's messages.

Some harnesses write tool output as ordinary text in the next user message;
the patterns that recognise it are here too.
"""
import re

import orjson


# Keys of a content block that carry transport details, not text.
MESSAGE_TRANSPORT_KEYS = frozenset(
    {
        "cache_control",
        "type",
        "is_error",
        "tool_use_id",
        "id",
        "index",
        "cache_creation",
        "citations",
        "signature",
        "annotations",
    }
)

# Content blocks holding a model's reasoning, which is never plain text.
REASONING_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})


def messages_list(prompt: str | None) -> list:
    """The input messages of a request body: the body itself when it is a
    JSON array, or its `messages` array; otherwise an empty list."""
    if prompt is None:
        return []
    try:
        body = orjson.loads(prompt)
    except orjson.JSONDecodeError:
        return []
    if isinstance(body, list):
        return body
    if isinstance(body, dict) and isinstance(body.get("messages"), list):
        return body["messages"]
    return []


def plain_text(message: object) -> str:
    """Every string in a message's `content` and `tool_calls`, joined with newlines.

    Transport keys and reasoning blocks are left out; nothing is normalised.
    """
    if not isinstance(message, dict):
        return ""
    pieces: list[str] = []
    collect_strings(message.get("content"), pieces)
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            collect_call_strings(call, pieces)
    return "\n".join(piece for piece in pieces if piece).strip()


def collect_strings(value: object, pieces: list[str]) -> None:
    """Append every string inside `value`, in order, skipping transport keys and reasoning."""
    if isinstance(value, str):
        pieces.append(value)
    elif isinstance(value, list):
        for element in value:
            collect_strings(element, pieces)
    elif isinstance(value, dict):
        if value.get("type") in REASONING_BLOCK_TYPES:
            return
        for key, element in value.items():
            if key not in MESSAGE_TRANSPORT_KEYS:
                collect_strings(element, pieces)


def collect_call_strings(call: object, pieces: list[str]) -> None:
    """Append a tool call's function name and arguments."""
    if not isinstance(call, dict):
        return
    function = call.get("function")
    if not isinstance(function, dict):
        return
    collect_strings(function.get("name"), pieces)
    collect_strings(function.get("arguments"), pieces)


def last_assistant_position(input_messages: list) -> int | None:
    """The position of the last message whose role is exactly `assistant`, or None."""
    for position in range(len(input_messages) - 1, -1, -1):
        message = input_messages[position]
        if isinstance(message, dict) and message.get("role") == "assistant":
            return position
    return None


def new_messages(input_messages: list) -> list[dict]:
    """The messages after the last assistant message; all of them when there is none."""
    position = last_assistant_position(input_messages)
    start = 0 if position is None else position + 1
    return [message for message in input_messages[start:] if isinstance(message, dict)]


# Pi's notice that a background job, started by a call some steps earlier, has finished.
BACKGROUND_JOB_NOTICE = re.compile(r"\*\*Background job \w+: bg_[0-9a-f]{6,}\*\*")
# Tool output that a harness writes as ordinary text in the next user message.
TOOL_OUTPUT_AS_TEXT = (
    re.compile(r"^\[[a-z_]+(?: for '.{0,2000}?')?\] Result:", re.M | re.S),  # cline and roo
    re.compile(r"^<returncode>\d+</returncode>", re.M),  # SWE-bench
    BACKGROUND_JOB_NOTICE,  # pi's background jobs
)


TOOL_DEFINITIONS = "tool_definitions"  # the role of the input's tool definitions entry
TOOL_DEFINITION_TRANSPORT_KEYS = frozenset({"cache_control", "signature"})  # keys removed from the tool definitions


def without_transport_keys(value: object) -> object:
    """The value with every `cache_control` and `signature` key removed, at any depth."""
    if isinstance(value, dict):
        return {key: without_transport_keys(item) for key, item in value.items() if key not in TOOL_DEFINITION_TRANSPORT_KEYS}
    if isinstance(value, list):
        return [without_transport_keys(item) for item in value]
    return value


def declared_tools(tools: str | None) -> list:
    """The request's tool definitions: its `tools` field when that is a non-empty JSON list, else none."""
    if not tools:
        return []
    try:
        value = orjson.loads(tools)
    except orjson.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def tool_name(tool: object) -> str | None:
    """A tool definition's name: `function.name` (the OpenAI chat shape), else `name` (the Anthropic and OpenAI
    Responses shapes); None when it has neither."""
    if not isinstance(tool, dict):
        return None
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return tool["name"] if isinstance(tool.get("name"), str) else None


def input_messages(prompt: str | None, system_prompt: str | None, tools: str | None = None) -> list:
    """The input's entries: the tool definitions when there are any, the top-level system prompt (JSON text, or
    None) when there is one, then the prompt's messages."""
    entries = []
    definitions = declared_tools(tools)
    if definitions:
        entries.append({"role": TOOL_DEFINITIONS, "content": without_transport_keys(definitions)})
    if system_prompt is not None:
        entries.append({"role": "system", "content": orjson.loads(system_prompt)})
    return entries + messages_list(prompt)
