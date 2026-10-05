"""The transitions step: each request after the first of its session, compared with the one before it.

The earlier input is the previous step's and the later input this step's, each
its tool definitions, system prompt and messages (`log_records/messages.py`). The
break is the first position at which their entry hashes differ. The
transition is `switch` when the two requests' models or providers differ;
`append` when every message of the earlier input is, unchanged, at the same
position of the later one; and `mutation` otherwise.

A mutation's cause is the first of these rules that holds:
1. `compaction`: a compaction note appears in the later input from the break
   on, and in none of the earlier input's messages from the break on;
2. `tool_definitions`: the entry at the break, in either input, is the tool
   definitions, so the tool list was changed, added or removed;
3. `system_prompt`: the earlier message at the break is the system prompt;
4. `dropped_turns`: the later message at the break is a later message of the
   earlier input, so turns were removed and the rest moved up;
5. `injection`: the earlier message at the break is a later message of the
   later input, so a turn was inserted ahead of it;
6. `tool_result`: both messages at the break are tool results;
7. `injection`: both are user messages, and the shorter one's text is the
   longer one's with a single span removed;
8. `other`: none of these.
Rules 1 to 6 read the digests; rule 7 reads the two messages' raw text.
Writes `transitions.parquet`.
"""

import dataclasses
import os
import pathlib

import pyarrow
from pyarrow import parquet

from export import parallel, paths
from export.log_records import messages, rendering
from export.sessions import extract

APPEND, MUTATION, SWITCH = "append", "mutation", "switch"

TRANSITIONS_SCHEMA = pyarrow.schema(
    [
        ("request_id", pyarrow.string()),
        ("transition", pyarrow.string()),
        ("cause", pyarrow.string()),  # null unless a mutation
        ("break_position", pyarrow.int32()),  # the first differing message; the earlier input's length on an append
    ]
)


@dataclasses.dataclass(frozen=True)
class InputFacts:
    """One request's input, as its digest holds it: per message, role, hash and compaction flag."""

    roles: list[str]
    hashes: list[int]
    notes: list[bool]


def break_position(earlier: InputFacts, later: InputFacts) -> int:
    """The first position at which the two inputs' messages differ, or the shorter one's length."""
    position = 0
    while position < min(len(earlier.hashes), len(later.hashes)) and earlier.hashes[position] == later.hashes[position]:
        position += 1
    return position


def cause_from_digests(earlier: InputFacts, later: InputFacts, at: int) -> str | None:
    """Rules 1 to 6, and `other` where rule 7 cannot apply; None when rule 7 must look at the text."""
    if any(later.notes[at:]) and not any(earlier.notes[at:]):
        return "compaction"
    if earlier.roles[at] == messages.TOOL_DEFINITIONS or later.roles[at:at + 1] == [messages.TOOL_DEFINITIONS]:
        return "tool_definitions"
    if earlier.roles[at] == "system":
        return "system_prompt"
    if at == len(later.hashes):
        return "other"
    if later.hashes[at] in earlier.hashes[at + 1:]:
        return "dropped_turns"
    if earlier.hashes[at] in later.hashes[at + 1:]:
        return "injection"
    if earlier.roles[at] == "tool" and later.roles[at] == "tool":
        return "tool_result"
    if earlier.roles[at] == "user" and later.roles[at] == "user":
        return None
    return "other"


def one_span_apart(first: str, second: str) -> bool:
    """Whether the shorter text is the longer one with a single span removed: a prefix of it, then a suffix."""
    shorter, longer = sorted((first, second), key=len)
    prefix = len(os.path.commonprefix([shorter, longer]))
    return longer.endswith(shorter[prefix:])


def read_input_facts(week_directory: pathlib.Path) -> dict[str, InputFacts]:
    """Every digest of the week, without its block hashes, by request id."""
    columns = ["request_id", "message_roles", "message_hashes", "compaction_notes"]
    facts = {}
    for row in parquet.read_table(week_directory / paths.DIGESTS_FILE, columns=columns).to_pylist():
        facts[row["request_id"]] = InputFacts(row["message_roles"], row["message_hashes"], row["compaction_notes"])
    return facts


def request_pairs(week_directory: pathlib.Path) -> list[tuple[dict, dict]]:
    """Each release request after the first of its session, with the request before it."""
    columns = ["request_id", "session_id", "step", "model_id", "provider"]
    requests = parquet.read_table(week_directory / paths.REQUESTS_FILE, columns=columns).to_pylist()
    ordered = sorted(requests, key=lambda request: (request["session_id"], request["step"]))
    return [(earlier, later) for earlier, later in zip(ordered, ordered[1:]) if earlier["session_id"] == later["session_id"]]


def transition_row(earlier: dict, later: dict, facts: dict[str, InputFacts]) -> dict:
    """The transition of one pair; a cause still to be decided on text is left as None."""
    earlier_facts, later_facts = facts[earlier["request_id"]], facts[later["request_id"]]
    at = break_position(earlier_facts, later_facts)
    row = {"request_id": later["request_id"], "transition": MUTATION, "cause": None, "break_position": at}
    if earlier["model_id"] != later["model_id"] or earlier["provider"] != later["provider"]:
        row["transition"] = SWITCH
    elif at == len(earlier_facts.hashes):
        row["transition"] = APPEND
    else:
        row["cause"] = cause_from_digests(earlier_facts, later_facts, at)
    return row


def rendered_message(prompt: str | None, system_prompt: str | None, tools: str | None, position: int) -> str:
    """The rendering of one input entry, counted as the digest counts them."""
    return rendering.render_message(messages.input_messages(prompt, system_prompt, tools)[position])


def decide_on_text(week_directory: pathlib.Path, rows: list[dict], earlier_of: dict[str, str]) -> None:
    """Rule 7 for every mutation left undecided: `injection` when one span apart, `other` otherwise."""
    undecided = [row for row in rows if row["transition"] == MUTATION and row["cause"] is None]
    needed = [row["request_id"] for row in undecided] + [earlier_of[row["request_id"]] for row in undecided]
    prompts = extract.prompts_of(week_directory / paths.EXTRACT_FILE, needed)
    tools = extract.column_of(week_directory / paths.EXTRACT_FILE, needed, "tools")
    system_table = parquet.read_table(week_directory / paths.SYSTEM_PROMPTS_FILE, columns=["request_id", "system"])
    system_prompts = dict(zip(system_table["request_id"].to_pylist(), system_table["system"].to_pylist()))
    for row in undecided:
        later_id, earlier_id = row["request_id"], earlier_of[row["request_id"]]
        at = row["break_position"]
        later_text = rendered_message(prompts[later_id], system_prompts.get(later_id), tools[later_id], at)
        earlier_text = rendered_message(prompts[earlier_id], system_prompts.get(earlier_id), tools[earlier_id], at)
        row["cause"] = "injection" if one_span_apart(earlier_text, later_text) else "other"


def write_week_transitions(week: paths.Week) -> dict[str, int]:
    """Write the week's `transitions.parquet`, and return the count of each transition and cause."""
    facts = read_input_facts(week.folder)
    pairs = request_pairs(week.folder)
    rows = [transition_row(earlier, later, facts) for earlier, later in pairs]
    decide_on_text(week.folder, rows, {later["request_id"]: earlier["request_id"] for earlier, later in pairs})
    parallel.write_rows(rows, TRANSITIONS_SCHEMA, week.folder / paths.TRANSITIONS_FILE)
    counts = {}
    for row in rows:
        key = row["cause"] or row["transition"]
        counts[key] = counts.get(key, 0) + 1
    return counts
