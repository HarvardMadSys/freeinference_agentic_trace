"""The tool results a request delivers, each reduced to its call id, its tokens and its outcome.

A request's fresh results are the ones after the last assistant message of
its input: the results of the calls the previous request made. They arrive
as OpenAI `tool` messages carrying a `tool_call_id`, or as `tool_result`
blocks, carrying a `tool_use_id`, inside a user message.

Two harnesses write their results into the user text instead, with no id:
Cline begins a text block with `[NAME] Result:` or `[NAME for '…'] Result:`,
and mini-swe-agent begins its text with `<returncode>N</returncode>`. These
are read as text results, in order, so the calls they answer can be found from
them (`calls.py`). The text of a result is read here and never kept.

A response's calls are matched to its next step's results together: each by
its own id first (without punctuation or a replayed `toolu` prefix), then by
its id without a `_<digits>` suffix, which some harnesses renumber. No result
answers two calls, since siblings `X_1`, `X_2`, … share that shorter id.
"""

import dataclasses
import re

import orjson


from export.log_records import call_ids, messages
from export.tool_calls import outcomes

SERIALISED_BLOCK_LIST = re.compile(r'^\s*\[\s*\{\s*"type"')  # a tool message's content, JSON-dumped as text
CLINE_RESULT = re.compile(r"\[([A-Za-z_][\w-]*)(?: for '.*?')?\] Result:", re.S)  # Cline's line naming the tool
RETURN_CODE_RESULT = re.compile(r"\s*<returncode>-?\d+</returncode>")  # mini-swe-agent's start of a command's output


@dataclasses.dataclass(frozen=True)
class ResultFacts:
    """What is kept of one tool result: the call it answers, its o200k tokens, and its outcome."""

    call_id: str
    tokens: int
    outcome: str


@dataclasses.dataclass(frozen=True)
class TextResult:
    """One result written into the user text: the tool Cline names (None for mini-swe-agent), its o200k tokens,
    and its outcome."""

    name: str | None
    tokens: int
    outcome: str


@dataclasses.dataclass(frozen=True)
class FreshResult:
    """One result with an id, as the input carries it: the call it answers, its text, and the harness's error flag."""

    call_id: str
    text: str
    is_error: object  # the `is_error` of a tool_result block; None for a tool message, which has none


@dataclasses.dataclass(frozen=True)
class CarriedResults:
    """What a request's new input brings back: its results with an id, its text results, and whether it holds
    Pi's notice of a finished background job."""

    with_ids: list[ResultFacts]
    as_text: list[TextResult]
    background_job: bool


def text_of_content(content: object) -> str:
    """A result's text: every string of its content, joined by line breaks; a JSON-dumped block list unpacked."""
    if isinstance(content, str) and SERIALISED_BLOCK_LIST.match(content):
        try:
            content = orjson.loads(content)
        except orjson.JSONDecodeError:
            pass
    pieces: list[str] = []
    messages.collect_strings(content, pieces)
    return "\n".join(piece for piece in pieces if piece)


def fresh_results(input_messages: list) -> list[FreshResult]:
    """The input's fresh results with an id, in order."""
    last_assistant = messages.last_assistant_position(input_messages)
    if last_assistant is None:
        return []
    found = []
    for message in input_messages[last_assistant + 1:]:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool" and isinstance(message.get("tool_call_id"), str):
            found.append(FreshResult(message["tool_call_id"], text_of_content(message.get("content")), None))
        elif message.get("role") == "user" and isinstance(message.get("content"), list):
            for block in message["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("tool_use_id"), str):
                    found.append(FreshResult(block["tool_use_id"], text_of_content(block.get("content")), block.get("is_error")))
    return found


def result_facts(prompt: str | None, encoding, outcome_rules: outcomes.OutcomeRules) -> list[ResultFacts]:
    """The facts of every fresh result with an id in a request's input."""
    facts = []
    for fresh in fresh_results(messages.messages_list(prompt)):
        facts.append(ResultFacts(fresh.call_id, len(encoding.encode_ordinary(fresh.text)),
                                 outcomes.outcome(fresh.text, fresh.is_error, outcome_rules)))
    return facts


def new_user_texts(input_messages: list) -> list[str]:
    """The texts of the user messages after the input's last assistant message: a string content as it is, and
    each text block of a list content."""
    last_assistant = messages.last_assistant_position(input_messages)
    texts = []
    for message in input_messages[(last_assistant + 1 if last_assistant is not None else 0):]:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                    texts.append(block["text"])
    return texts


def text_results(input_messages: list, encoding, outcome_rules: outcomes.OutcomeRules) -> list[TextResult]:
    """The results written into the new user text, in order: each Cline result block, and a mini-swe-agent text."""
    found = []
    for text in new_user_texts(input_messages):
        cline = CLINE_RESULT.match(text)
        if cline is None and RETURN_CODE_RESULT.match(text) is None:
            continue
        name = cline.group(1) if cline else None
        found.append(TextResult(name, len(encoding.encode_ordinary(text)), outcomes.outcome(text, None, outcome_rules)))
    return found


def carried_results(prompt: str | None, encoding, outcome_rules: outcomes.OutcomeRules) -> CarriedResults:
    """Every result a request's new input brings back, with an id or as text, and its background-job notice."""
    input_messages = messages.messages_list(prompt)
    background_job = any(messages.BACKGROUND_JOB_NOTICE.search(text) for text in new_user_texts(input_messages))
    return CarriedResults(result_facts(prompt, encoding, outcome_rules), text_results(input_messages, encoding, outcome_rules),
                          background_job)


def first_untaken(fresh: list[ResultFacts], taken: set[int], wanted_keys: set[str], keys_of_result) -> int | None:
    """The position of the first result not yet taken whose id has one of the wanted keys; None when none has."""
    if not wanted_keys:
        return None
    for position, facts in enumerate(fresh):
        if position not in taken and keys_of_result(facts.call_id) & wanted_keys:
            return position
    return None


def match_results(ids: list, fresh: list[ResultFacts]) -> list[ResultFacts | None]:
    """The result of each call of one response, in call order: first by its own id, then, for a call still
    without one, by its id without a `_<digits>` suffix; no result is given to two calls."""
    chosen = [None] * len(ids)
    taken = set()
    for index, call_id in enumerate(ids):
        chosen[index] = first_untaken(fresh, taken, call_ids.own_keys(call_id), call_ids.own_keys)
        if chosen[index] is not None:
            taken.add(chosen[index])
    for index, call_id in enumerate(ids):
        if chosen[index] is None:
            renumbered = call_ids.emitted_keys(call_id) - call_ids.own_keys(call_id)
            chosen[index] = first_untaken(fresh, taken, renumbered, call_ids.replayed_keys)
            if chosen[index] is not None:
                taken.add(chosen[index])
    return [None if position is None else fresh[position] for position in chosen]
