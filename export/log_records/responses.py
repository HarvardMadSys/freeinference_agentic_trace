"""A response body's visible text, its tool calls, and the reason it says it ended.

Two response shapes are read: OpenAI chat completions (`choices`) and
Anthropic messages (`stop_reason`, or `type: "message"`). Anything else has no
visible text and no calls. Reasoning is never visible text.
"""

import dataclasses
import json
import re

from export.log_records import fields

OPENAI = "openai"  # a call in a chat completion's `tool_calls`
ANTHROPIC = "anthropic"  # a `tool_use` block of a message
TEXT = "text"  # a call written inside the visible text

# Tool calls some models write inside their visible text, tried in this order.
TEXT_CALL_OPENERS = ("<function=", '<invoke name="')
TEXT_CALL_GRAMMARS = (
    (
        re.compile(r"<function=([A-Za-z0-9_.\-]+)\s*>(.*?)(?:</function>|\Z)", re.S),
        re.compile(r"<parameter=([A-Za-z0-9_.\-]+)\s*>(.*?)</parameter>", re.S),
    ),
    (
        re.compile(r'<invoke name="([^"]+)"\s*>(.*?)(?:</invoke>|\Z)', re.S),
        re.compile(r'<parameter name="([^"]+)"\s*>(.*?)</parameter>', re.S),
    ),
)
# A child element of a call written as XML: one argument, `<key>value</key>`.
CHILD_ELEMENT = re.compile(r"<([A-Za-z_][\w-]*)>(.*?)</\1>", re.S)
# A fenced block: its info string, then its text up to the closing fence.
FENCED_BLOCK = re.compile(r"```[^\n`]*\n(.*?)\n[ \t]*```", re.S)
FENCED_COMMAND_NAME = "bash"  # the name of the one command a fenced block holds


@dataclasses.dataclass(frozen=True)
class Call:
    """One tool call: its id and name as written (possibly None), and its arguments as text."""

    id: object
    name: object
    arguments_text: str | None
    # How the call was written: `openai`, `anthropic` or `text`. Two calls that
    # say the same thing are equal whichever way they were written.
    call_format: str = dataclasses.field(default="", compare=False)


@dataclasses.dataclass(frozen=True)
class Response:
    """What a response visibly said, the tool calls it made, in order, and the reason it stated for ending."""

    visible_text: str | None
    calls: tuple[Call, ...]
    stated_reason: str | None = None  # an OpenAI first choice's `finish_reason`, or an Anthropic `stop_reason`


NO_RESPONSE = Response(visible_text=None, calls=())


def read_response(response: str | None) -> Response:
    """The visible text and calls of a response body."""
    if response is None:
        return NO_RESPONSE
    body = fields.parse_object(response)
    if body is None:
        return NO_RESPONSE
    if "choices" in body:
        visible_text, calls = read_openai(body)
        stated_reason = openai_finish_reason(body)
    elif "stop_reason" in body or body.get("type") == "message":
        visible_text, calls = read_anthropic(body)
        stated_reason = body.get("stop_reason")
    else:
        return NO_RESPONSE
    if not calls and visible_text:
        calls = text_calls(visible_text)
    stated_reason = stated_reason if isinstance(stated_reason, str) else None
    return Response(visible_text=visible_text, calls=calls, stated_reason=stated_reason)


def openai_finish_reason(body: dict) -> object:
    """The first choice's `finish_reason`, as written; None without one."""
    choices = body["choices"]
    if not isinstance(choices, list) or not choices:
        return None
    return as_object(choices[0]).get("finish_reason")


def read_openai(body: dict) -> tuple[str | None, tuple[Call, ...]]:
    """An OpenAI chat completion: the first choice's text and calls."""
    choices = body["choices"]
    if not isinstance(choices, list) or not choices:
        return None, ()
    message = as_object(as_object(choices[0]).get("message"))
    content = message.get("content")
    visible_text = content if isinstance(content, str) else None
    calls = openai_calls(message)
    if not calls:
        calls = calls_of_first_choice_with_calls(choices)
    return visible_text, calls


def calls_of_first_choice_with_calls(choices: list) -> tuple[Call, ...]:
    """Some providers put the calls on a later choice: the calls of the first choice that has any."""
    for choice in choices:
        message = as_object(as_object(choice).get("message"))
        if message.get("tool_calls"):
            return openai_calls(message)
    return ()


def openai_calls(message: dict) -> tuple[Call, ...]:
    """The calls in an OpenAI message's `tool_calls`."""
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return ()
    calls = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = as_object(tool_call.get("function"))
        calls.append(Call(tool_call.get("id"), function.get("name"), arguments_text(function.get("arguments")), OPENAI))
    return tuple(calls)


def as_object(value: object) -> dict:
    """The value when it is a JSON object, else an empty one."""
    if isinstance(value, dict):
        return value
    return {}


def read_anthropic(body: dict) -> tuple[str | None, tuple[Call, ...]]:
    """An Anthropic message: its text blocks joined with no separator, and its `tool_use` blocks."""
    content = body.get("content")
    if not isinstance(content, list):
        return None, ()
    texts = []
    calls = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
        if block.get("type") == "tool_use":
            calls.append(Call(block.get("id"), block.get("name"), arguments_text(block.get("input")), ANTHROPIC))
    visible_text = "".join(texts) if texts else None
    return visible_text, tuple(calls)


def arguments_text(value: object) -> str | None:
    """A call's arguments as text: a string as it is, None as None, anything else as JSON."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def text_calls(visible_text: str) -> tuple[Call, ...]:
    """Tool calls written inside the visible text, such as `<function=read_file>...</function>`."""
    if not any(opener in visible_text for opener in TEXT_CALL_OPENERS):
        return ()
    calls = []
    for call_pattern, parameter_pattern in TEXT_CALL_GRAMMARS:
        for call_match in call_pattern.finditer(visible_text):
            arguments = {}
            for parameter_match in parameter_pattern.finditer(call_match.group(2)):
                arguments[parameter_match.group(1)] = parameter_match.group(2).strip()
            calls.append(Call(None, call_match.group(1), json.dumps(arguments, ensure_ascii=False), TEXT))
    return tuple(calls)


def named_element_call(visible_text: str, name: str, occurrence: int) -> Call | None:
    """The text's `occurrence`-th element `<name>…</name>`, from 0, as a call written in the text: its child
    elements are its arguments. An element left open runs to the end of the text. None when there is no such
    element."""
    pattern = re.compile(rf"<{re.escape(name)}>(.*?)(?:</{re.escape(name)}>|\Z)", re.S)
    elements = pattern.findall(visible_text)
    if occurrence >= len(elements):
        return None
    arguments = {key: value.strip() for key, value in CHILD_ELEMENT.findall(elements[occurrence])}
    return Call(None, name, json.dumps(arguments, ensure_ascii=False), TEXT)


def fenced_command_call(visible_text: str) -> Call | None:
    """The text's one fenced block as one command, `bash` with its text as `command`; None unless the text holds
    exactly one fenced block."""
    blocks = FENCED_BLOCK.findall(visible_text)
    if len(blocks) != 1:
        return None
    arguments = {"command": blocks[0].strip()}
    return Call(None, FENCED_COMMAND_NAME, json.dumps(arguments, ensure_ascii=False), TEXT)
