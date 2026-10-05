"""A call's canonical name, family, shell purpose and category, by the rules of the tool tables.

`export/configs/tool_names.toml` folds the spellings of a tool into one canonical name;
`export/configs/tool_categories.toml` gives each canonical name a family, each shell
program a purpose, and each family and purpose the category it is counted in.
"""

import dataclasses
import pathlib
import re
import tomllib

from release import schema
from export.tool_calls import commands

UNKNOWN_TOOL = "unknown"  # the canonical name of a call with no name
WORD_SEPARATOR = re.compile(r"[^a-z0-9]+")
SHELL_FAMILY = "shell"
SUBAGENT_FAMILY = "subagent"
ASK_USER_FAMILY = "ask_user"
MCP_FAMILY = "mcp"
OTHER_FAMILY = "other"  # the family of a native call no table lists
OTHER_PURPOSE = "other"  # the purpose of a shell program no purpose list names
WAITING = "waiting"
FILES = "files"
INTERPRETER = "interpreter"
PACKAGES = "packages"
POWERSHELL_COMMAND = re.compile(r"^[a-z]+-([a-z]+)$")  # `get-content`: a verb, a dash, a noun
PIPE = "|"


@dataclasses.dataclass(frozen=True)
class ToolTables:
    """The two tool tables, loaded."""

    aliases: dict[str, str]  # spelling -> canonical name
    keyword_rules: list[tuple[str, list[list[str]]]]  # canonical name and its keywords as words, in order
    spawn_tools: frozenset[str]
    ask_user_tools: frozenset[str]
    family_of_name: dict[str, str]  # canonical name -> family
    mcp_markers: tuple[str, ...]
    purposes: dict[str, frozenset[str]]  # purpose -> its programs
    purpose_order: list[str]
    setup_programs: frozenset[str]
    plumbing_programs: frozenset[str]
    run_subcommands: frozenset[str]
    powershell_file_nouns: frozenset[str]
    categories: dict[str, str]  # family or purpose -> category


def words_of(name: str) -> list[str]:
    """The words of a lower-case name, split at any character that is not a letter or a digit."""
    return [word for word in WORD_SEPARATOR.split(name) if word]


def load_tables(names_file: pathlib.Path, categories_file: pathlib.Path) -> ToolTables:
    """Both tool tables."""
    with open(names_file, "rb") as handle:
        names = tomllib.load(handle)
    with open(categories_file, "rb") as handle:
        categories = tomllib.load(handle)
    shell_rules = categories["shell_rules"]
    return ToolTables(
        aliases={spelling: canonical for canonical, spellings in names["aliases"].items() for spelling in spellings},
        keyword_rules=[(canonical, [words_of(keyword) for keyword in keywords])
                       for canonical, keywords in names["keywords"].items()],
        spawn_tools=frozenset(categories["spawn"]["tools"]),
        ask_user_tools=frozenset(categories["ask_user"]["tools"]),
        family_of_name={name: family for family, members in categories["tool_families"].items() for name in members},
        mcp_markers=tuple(categories["mcp"]["name_contains"]),
        purposes={purpose: frozenset(programs) for purpose, programs in categories["shell_purposes"].items()},
        purpose_order=list(shell_rules["purpose_order"]),
        setup_programs=frozenset(shell_rules["setup_programs"]),
        plumbing_programs=frozenset(shell_rules["plumbing_programs"]),
        run_subcommands=frozenset(shell_rules["run_subcommands"]),
        powershell_file_nouns=frozenset(shell_rules["powershell_file_nouns"]),
        categories=dict(categories["categories"]),
    )


def starts_a_word(words: list[str], keyword: list[str]) -> bool:
    """Whether the keyword's words appear in `words` in a row, the last as the start of a word."""
    size = len(keyword)
    for start in range(len(words) - size + 1):
        window = words[start:start + size]
        if window[:-1] == keyword[:-1] and window[-1].startswith(keyword[-1]):
            return True
    return False


def canonical_name(raw_name: object, tool_tables: ToolTables) -> str:
    """The raw name folded to its canonical name; a name no rule matches is kept, cleaned."""
    if not isinstance(raw_name, str) or not raw_name.strip():
        return UNKNOWN_TOOL
    cleaned = raw_name.strip().lower()
    if cleaned in tool_tables.aliases:
        return tool_tables.aliases[cleaned]
    words = words_of(cleaned)
    for canonical, keywords in tool_tables.keyword_rules:
        if any(starts_a_word(words, keyword) for keyword in keywords):
            return canonical
    return cleaned


def family(raw_name: object, canonical: str, tool_tables: ToolTables) -> str:
    """The call's family: the spawn list first, then the family map, the ask-user list and the MCP test."""
    lowered = raw_name.lower() if isinstance(raw_name, str) else ""
    if lowered in tool_tables.spawn_tools:
        return SUBAGENT_FAMILY
    if canonical in tool_tables.family_of_name:
        return tool_tables.family_of_name[canonical]
    if lowered in tool_tables.ask_user_tools:
        return ASK_USER_FAMILY
    if any(marker in canonical for marker in tool_tables.mcp_markers):
        return MCP_FAMILY
    return OTHER_FAMILY


def first_word(label: str) -> str:
    """A label's first word, lower-cased."""
    return label.split()[0].lower() if label.split() else ""


def deciding_program(programs: list[str], joined_by: list[str], tool_tables: ToolTables) -> int | None:
    """The program that decides a command's purpose: in the last pipeline that has one, its first program
    that is neither setup nor plumbing. None when every program is setup or plumbing."""
    pipelines: list[list[int]] = []
    for index, operator in enumerate(joined_by):
        if index == 0 or operator != PIPE:
            pipelines.append([])
        pipelines[-1].append(index)
    ignored = tool_tables.setup_programs | tool_tables.plumbing_programs
    for pipeline in reversed(pipelines):
        for index in pipeline:
            if first_word(programs[index]) not in ignored:
                return index
    return None


def purpose_of_label(label: str, tool_tables: ToolTables) -> str:
    """The purpose of one program label: the first purpose in order that lists a word of it."""
    if label.endswith(commands.CODE_LABEL_SUFFIX):
        return INTERPRETER
    words = [word.lower() for word in label.split()]
    noun = POWERSHELL_COMMAND.match(words[0]) if words else None
    for purpose in tool_tables.purpose_order:
        if any(word in tool_tables.purposes.get(purpose, frozenset()) for word in words):
            if purpose == PACKAGES and len(words) > 1 and words[1] in tool_tables.run_subcommands:
                return INTERPRETER
            return purpose
    if noun and noun.group(1) in tool_tables.powershell_file_nouns:
        return FILES
    return OTHER_PURPOSE


def shell_purpose(programs: list[str], deciding: int | None, backgrounded: bool, skeleton: str,
                  tool_tables: ToolTables) -> str | None:
    """A shell call's purpose; None when no program was read.

    A call waits when a command runs in the background or a word of its
    skeleton is on the waiting list; the skeleton keeps wrappers such as
    `nohup`, which are not programs.
    """
    if not programs:
        return None
    waiting_programs = tool_tables.purposes.get(WAITING, frozenset())
    if backgrounded or any(word.lower() in waiting_programs for word in skeleton.split()):
        return WAITING
    if deciding is None:
        return FILES
    return purpose_of_label(programs[deciding], tool_tables)


def category(call_family: str, purpose: str | None, tool_tables: ToolTables) -> str:
    """The category a call is counted in: its family's, or for a shell call its purpose's."""
    key = purpose if call_family == SHELL_FAMILY else call_family
    if key is None:
        return schema.OTHER_CATEGORY
    return tool_tables.categories.get(key, schema.OTHER_CATEGORY)
