"""An input entry rendered as the text the model read, and the role it counts under.

Each entry is rendered as the harness sent it, replayed reasoning included.
The only two changes to the text: a leading billing header is removed, and
base64 payloads become `<base64>`. The role is one of the release's: system,
user, assistant, tool, or the tool definitions.
"""
import json
import re

from export.log_records import messages


BILLING_HEADER = re.compile(r"^\s*x-anthropic-billing-header:\s*(?:[\w.-]+=[^;\n]*;\s*)+")
BASE64_PAYLOAD = re.compile(r"(data:[^;]*;base64,)[A-Za-z0-9+/=]{200,}")

# Blocks that carry no text a model reads, rendered as `<type>`.
PLACEHOLDER_BLOCK_TYPES = frozenset(
    {"redacted_thinking", "image", "image_url", "document", "file", "audio", "input_audio"}
)
TOOL_RESULT_BLOCK_TYPES = frozenset({"tool_result", "tool-result"})
SYSTEM_ROLES = frozenset({"system", "developer"})


def compact_json(value: object) -> str:
    """JSON with `,` and `:` separators, keys in their original order, non-ASCII kept."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def without_billing_header(text: str) -> str:
    """The text without claude-code's leading accounting line."""
    return BILLING_HEADER.sub("", text, count=1)


def render_block(block: object) -> str:
    """One content block as text, by its type."""
    if isinstance(block, str):
        return without_billing_header(block)
    if not isinstance(block, dict):
        return compact_json(block)
    block_type = block.get("type")
    if block_type == "text":
        return without_billing_header(block.get("text") or "")
    if block_type == "tool_use":
        return f"{block.get('name') or ''} {compact_json(block.get('input'))}"
    if block_type in TOOL_RESULT_BLOCK_TYPES:
        return render_content(block.get("content"))
    if block_type == "thinking":
        return block.get("thinking") or ""
    if block_type in PLACEHOLDER_BLOCK_TYPES:
        return f"<{block_type}>"
    return compact_json({key: value for key, value in block.items() if key != "cache_control"})


def render_content(content: object) -> str:
    """A message's content: a string as it is, a list block by block, joined with newlines."""
    if content is None:
        return ""
    if isinstance(content, str):
        return without_billing_header(content)
    if isinstance(content, list):
        return "\n".join(render_block(block) for block in content)
    return compact_json(content)


def render_message(message: object) -> str:
    """The message as one text: role, reasoning, content and tool calls, joined with newlines. The tool
    definitions entry is its JSON alone."""
    if not isinstance(message, dict):
        return compact_json(message)
    if message.get("role") == messages.TOOL_DEFINITIONS:
        return compact_json(message.get("content"))
    pieces = [str(message.get("role") or "")]
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        pieces.append(reasoning)
    content = render_content(message.get("content"))
    if content:
        pieces.append(content)
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            pieces.append(render_call(call))
    return BASE64_PAYLOAD.sub(r"\1<base64>", "\n".join(pieces))


def render_call(call: object) -> str:
    """A tool call on an input message: its name, a space, and its arguments as sent."""
    if not isinstance(call, dict):
        return compact_json(call)
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        arguments = compact_json(arguments)
    return f"{function.get('name') or ''} {arguments}"


def role_of(message: object) -> str:
    """The message's role in the release: system, user, assistant, tool or other; the input's tool definitions
    entry is `tool_definitions`."""
    if not isinstance(message, dict):
        return "other"
    role = message.get("role")
    if role == messages.TOOL_DEFINITIONS:
        return messages.TOOL_DEFINITIONS
    if role in SYSTEM_ROLES:
        return "system"
    if role == "assistant":
        return "assistant"
    if role == "tool":
        return "tool"
    if role == "user":
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") in TOOL_RESULT_BLOCK_TYPES for block in content
        ):
            return "tool"
        return "user"
    return "other"
