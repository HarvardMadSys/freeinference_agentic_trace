"""The public list of tool words: which tool names, program labels, argument keys, skeleton words, subagent types and
stand-ins may ship.

A word is public when at least three distinct accounts used it, over every
week's tool calls. For tool names, an account also uses a name when it
declared a tool of that name in a request's tool definitions, read from the
week's digests. There are six lists: tool names, lower-cased; program labels;
argument keys; the words of command skeletons; the values of spawn calls'
`subagent_type` argument; and the typed stand-ins of hidden programs
(`<packages>`, `<script>.sh`, see `stand_ins.py`).
`configs/public_tool_words.toml` holds each list; its `[added]` table
holds words made public by hand, which a rebuild keeps. A tool name ships
lower-cased when it is public, else as `other`; its category is decided from
the full name, so it does not depend on the list. The stand-ins are counted
after the words, because which stand-in a program takes depends on which
words are public.
"""

import collections
import dataclasses
import functools
import pathlib
import tomllib

from pyarrow import parquet

from release import schema
from export import paths
from export.privacy import stand_ins
from export.tool_calls import arguments, categories

MIN_ACCOUNTS = 3  # a word or stand-in used by fewer accounts could name one of them, so it does not ship
WORD_LISTS = ("names", "programs", "argument_keys", "skeleton_words", "subagent_types")  # counted in the first pass
LISTS = WORD_LISTS + ("stand_ins",)
# The committed list's header: what the lists are, why the file is committed, and what a change to it forces.
HEADER = (
    f"# The public tool words: what a tool call may show as written, since at least {MIN_ACCOUNTS} accounts used it,",
    "# counted over every week of the release, and for a tool name over calls and declarations alike. The lists:",
    "# names (tool names), programs (the program labels of shell calls), argument_keys, skeleton_words (the command",
    "# words of argument skeletons), subagent_types (the subagent_type values of spawn calls) and stand_ins (the typed",
    "# stand-ins that replace hidden programs and words).",
    "# Written by `python -m export tool-words` from the private week folders, which a reader of the release does not",
    "# have, so the file is committed. A change to it changes what every released week may show: release every week",
    "# again (`python -m export release --all-weeks`), then `make figures`. [added] holds words made public by hand,",
    "# which a rebuild keeps.",
)
READ_COLUMNS = ["request_id", "raw_name", "programs_full", "arguments_skeleton_full", "arguments"]


@dataclasses.dataclass(frozen=True)
class PublicWords:
    """The six public lists."""

    names: frozenset[str]
    programs: frozenset[str]
    argument_keys: frozenset[str]
    skeleton_words: frozenset[str]
    stand_ins: frozenset[str] = frozenset()
    subagent_types: frozenset[str] = frozenset()


NO_WORDS = PublicWords(frozenset(), frozenset(), frozenset(), frozenset())


def load_public_words(path: pathlib.Path) -> PublicWords:
    """The committed public lists, each with its words added by hand; empty lists when there is no file yet."""
    if not path.exists():
        return NO_WORDS
    with open(path, "rb") as handle:
        table = tomllib.load(handle)
    added = table.get("added", {})
    lists = {name: frozenset(table.get(name, {})) | frozenset(added.get(name, [])) for name in LISTS}
    for name, entries in lists.items():
        if any(arguments.holds_ip_address(entry) for entry in entries):
            raise ValueError(f"{path}: the {name} list holds an IP address; no IP address may ship")
    return PublicWords(**lists)


def listed_name(raw_name: str | None) -> str | None:
    """A tool name as the `names` list holds it: lower-cased; None when it is missing or blank."""
    if raw_name is None or not raw_name.strip():
        return None
    return raw_name.lower()


def released_name(raw_name: str | None, words: PublicWords) -> str:
    """A tool's name as it ships, of a call or of a declared tool: lower-cased when it is public, else `other`."""
    name = listed_name(raw_name)
    return name if name in words.names else schema.OTHER_WORD


def words_of_call(call: dict) -> dict[str, set[str]]:
    """The words one call used, per word list."""
    skeleton = call["arguments_skeleton_full"]
    used = {"names": {listed_name(call["raw_name"])} - {None}, "programs": set(call["programs_full"] or []),
            "argument_keys": {key for key, _ in arguments.skeleton_pairs(skeleton)}, "skeleton_words": set(),
            "subagent_types": set()}
    subagent_type = arguments.subagent_type(call["arguments"])
    if subagent_type is not None:
        used["subagent_types"].add(subagent_type)
    for command in arguments.command_values(skeleton):
        used["skeleton_words"].update(arguments.skeleton_words(command))
    return used


def stand_ins_of_call(call: dict, words: PublicWords, tool_tables: categories.ToolTables) -> dict[str, set[str]]:
    """The typed stand-ins one call's hidden programs would take: of each hidden label with no public prefix, and
    of each hidden skeleton word in a program's place."""
    candidates = set()
    for label in call["programs_full"] or []:
        if label not in words.programs and stand_ins.public_prefix(label, words.programs) is None:
            candidates.add(stand_ins.typed_candidate(label, tool_tables))
    for command in arguments.command_values(call["arguments_skeleton_full"]):
        command_words = command.split()
        for position in stand_ins.program_places(command_words):
            if command_words[position] not in words.skeleton_words:
                candidates.add(stand_ins.typed_candidate(command_words[position], tool_tables))
    candidates.discard(None)
    return {"stand_ins": candidates}


def accounts_per_entry(week_directories: list[pathlib.Path], entries_of_call) -> dict[str, dict[str, set[str]]]:
    """For each list that `entries_of_call` fills, the accounts that used each entry, over every week."""
    accounts = collections.defaultdict(lambda: collections.defaultdict(set))
    for week_directory in week_directories:
        users = parquet.read_table(week_directory / paths.REQUESTS_FILE, columns=["request_id", "user_id"])
        user_of = dict(zip(users["request_id"].to_pylist(), users["user_id"].to_pylist()))
        tool_calls = parquet.ParquetFile(week_directory / paths.TOOL_CALLS_FILE)
        for row_group in range(tool_calls.num_row_groups):
            for call in tool_calls.read_row_group(row_group, columns=READ_COLUMNS).to_pylist():
                for name, used in entries_of_call(call).items():
                    for entry in used:
                        accounts[name][entry].add(user_of[call["request_id"]])
    return accounts


def declared_name_accounts(week_directories: list[pathlib.Path]) -> dict[str, set[str]]:
    """The accounts that declared a tool of each name, lower-cased, over every week's digests."""
    accounts = collections.defaultdict(set)
    for week_directory in week_directories:
        users = parquet.read_table(week_directory / paths.REQUESTS_FILE, columns=["request_id", "user_id"])
        user_of = dict(zip(users["request_id"].to_pylist(), users["user_id"].to_pylist()))
        digests = parquet.ParquetFile(week_directory / paths.DIGESTS_FILE)
        for row_group in range(digests.num_row_groups):
            table = digests.read_row_group(row_group, columns=["request_id", "tool_definition_names"])
            for request_id, names in zip(table["request_id"].to_pylist(), table["tool_definition_names"].to_pylist()):
                for raw_name in names or []:
                    name = listed_name(raw_name)
                    if name is not None:
                        accounts[name].add(user_of[request_id])
    return accounts


def is_public(entry: str, users: set[str]) -> bool:
    """Whether an entry ships: at least MIN_ACCOUNTS accounts used it, and it holds no IP address, however many did."""
    return len(users) >= MIN_ACCOUNTS and not arguments.holds_ip_address(entry)


def public_entries(accounts: dict[str, set[str]], added: list[str]) -> frozenset[str]:
    """The entries that ship, and those added by hand."""
    return frozenset(entry for entry, users in accounts.items() if is_public(entry, users)) | frozenset(added)


def toml_key(word: str) -> str:
    """A word as a quoted TOML key."""
    return '"' + word.replace("\\", "\\\\").replace('"', '\\"') + '"'


def public_words_text(accounts: dict[str, dict[str, set[str]]], added: dict[str, list[str]]) -> str:
    """The public list file: each list's entries used by at least MIN_ACCOUNTS accounts, sorted."""
    lines = list(HEADER) + [""]
    for name in LISTS:
        lines.append(f"{name} = [")
        for word in sorted(accounts[name]):
            if is_public(word, accounts[name][word]):
                lines.append(f"    {toml_key(word)},")
        lines.append("]")
        lines.append("")
    lines.append("[added]")
    for name in LISTS:
        lines.append(f"{name} = [" + ", ".join(toml_key(word) for word in sorted(added.get(name, []))) + "]")
    return "\n".join(lines) + "\n"


def write_public_words(week_directories: list[pathlib.Path], path: pathlib.Path,
                       tool_tables: categories.ToolTables) -> dict[str, int]:
    """Rebuild the public list from every week's tool calls and declared tools, keeping the entries added by hand;
    return list sizes."""
    added = {}
    if path.exists():
        with open(path, "rb") as handle:
            added = tomllib.load(handle).get("added", {})
    accounts = accounts_per_entry(week_directories, words_of_call)
    for name, users in declared_name_accounts(week_directories).items():
        accounts["names"][name] |= users
    words = PublicWords(**{name: public_entries(accounts[name], added.get(name, [])) for name in WORD_LISTS})
    counting = functools.partial(stand_ins_of_call, words=words, tool_tables=tool_tables)
    accounts["stand_ins"] = accounts_per_entry(week_directories, counting)["stand_ins"]
    path.write_text(public_words_text(accounts, added))
    return {name: len(entries) for name, entries in vars(load_public_words(path)).items()}


