"""Which of the paper's harnesses sent a request, from `export/configs/harnesses.toml`.

A request's harness is the table entry whose self-identifying phrase appears
earliest in the request's intro window; failing that, the first entry with a
user-agent substring found in the user agent; otherwise `other-agent`. Only
the paper's harnesses are listed.
"""

import dataclasses
import pathlib
import tomllib

from export.log_records import messages

INTRO_ROLES = frozenset({"system", "user"})  # only these messages introduce a harness
INTRO_MAX_MESSAGES = 6  # a harness is recognised from the first 6 input entries only
INTRO_MAX_CHARACTERS = 20_000  # and from the first 20,000 characters of their text
UNRECOGNISED_HARNESS = "other-agent"  # the harness of a request no phrase or user agent names
BENCHMARK_HARNESS = "eval:swe-bench"  # the one benchmark harness: SWE-bench runs under mini-swe-agent


class HarnessTableError(Exception):
    """The harness table is malformed."""


@dataclasses.dataclass(frozen=True)
class Harness:
    """One table entry: the harness's name, its phrases and its user-agent substrings, all lower-case."""

    name: str
    phrases: tuple[str, ...]
    user_agents: tuple[str, ...]


def load_harness_table(path: pathlib.Path) -> tuple[Harness, ...]:
    """The table's entries, in the order written."""
    with open(path, "rb") as handle:
        table = tomllib.load(handle)
    harnesses = []
    for name, entry in table.items():
        if name == "version":
            continue
        harness = Harness(
            name=name,
            phrases=tuple(entry.get("phrases", [])),
            user_agents=tuple(entry.get("user_agent", [])),
        )
        for text in harness.phrases + harness.user_agents:
            if text != text.lower():
                raise HarnessTableError(f"{name}: {text!r} is not lower-case")
        harnesses.append(harness)
    return tuple(harnesses)


def intro_window(input_messages: list) -> str:
    """The lower-cased plain text of the system and user messages among the first entries."""
    texts = []
    for message in input_messages[: INTRO_MAX_MESSAGES]:
        if isinstance(message, dict) and message.get("role") in INTRO_ROLES:
            texts.append(messages.plain_text(message))
    return "\n".join(texts)[: INTRO_MAX_CHARACTERS].lower()


def harness_name(input_messages: list, user_agent: str | None, harnesses: tuple[Harness, ...]) -> str:
    """The harness of a request with these input messages and this user agent."""
    name = harness_by_earliest_phrase(intro_window(input_messages), harnesses)
    if name is None:
        name = harness_by_user_agent(user_agent, harnesses)
    if name is None:
        name = UNRECOGNISED_HARNESS
    return name


def harness_by_earliest_phrase(window: str, harnesses: tuple[Harness, ...]) -> str | None:
    """The entry whose phrase starts earliest in the window; on a tie, the entry listed first."""
    earliest_position = None
    earliest_name = None
    for harness in harnesses:
        for phrase in harness.phrases:
            position = window.find(phrase)
            if position == -1:
                continue
            if earliest_position is None or position < earliest_position:
                earliest_position = position
                earliest_name = harness.name
    return earliest_name


def harness_by_user_agent(user_agent: str | None, harnesses: tuple[Harness, ...]) -> str | None:
    """The first entry with a user-agent substring found in the lower-cased user agent."""
    if not user_agent:
        return None
    lowered_user_agent = user_agent.lower()
    for harness in harnesses:
        for substring in harness.user_agents:
            if substring in lowered_user_agent:
                return harness.name
    return None
