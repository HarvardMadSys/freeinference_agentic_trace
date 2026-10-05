"""A tool call's columns as they ship: its name, its program labels and its argument skeleton, anonymised by the public
word list. The release computes them from the full columns when it builds a week; no private file stores them.

A tool name ships lower-cased when it is public, else as `other`. A program
label ships when public, else as its stand-in. A skeleton keeps its public
keys and replaces the others by `<key>`; a hidden command word ships as its
stand-in; a spawn call's `subagent_type` value ships when it is public. The
same rules, read backwards, say whether a shipped value is allowed.
"""

import json

import pyarrow

from export.privacy import public_words, stand_ins
from export.tool_calls import arguments, categories

KEY_MARK = "<key>"  # a hidden argument key
ANONYMISED_COLUMNS = [
    ("name", pyarrow.string()),
    ("programs", pyarrow.list_(pyarrow.string())),
    ("arguments_skeleton", pyarrow.large_string()),
]


def anonymised_label(label: str, words: public_words.PublicWords, tool_tables: categories.ToolTables) -> str:
    """A program label as it ships: itself when public, else its stand-in."""
    if label in words.programs:
        return label
    return stand_ins.label_stand_in(label, words.programs, words.stand_ins, tool_tables)


def with_subagent_type(skeleton: str, value: str | None, public_types: frozenset[str]) -> str:
    """The skeleton with its `subagent_type` placeholder replaced by the call's own value, when that value is public."""
    if value is None or value not in public_types or skeleton == arguments.UNPARSED:
        return skeleton
    pairs = []
    for key, placeholder in arguments.skeleton_pairs(skeleton):
        pairs.append((key, value if key == arguments.SUBAGENT_TYPE_KEY and placeholder == arguments.STRING_MARK else placeholder))
    return arguments.pairs_text(pairs)


def public_command(command_skeleton: str, public_skeleton_words: frozenset[str], public_stand_ins: frozenset[str],
                   tool_tables: categories.ToolTables) -> str:
    """A command skeleton with every word that is not a placeholder, an operator or public replaced by its
    stand-in (see `stand_ins.py`)."""
    words = command_skeleton.split()
    places = stand_ins.program_places(words)
    shipped = []
    for position, word in enumerate(words):
        if word in arguments.PLACEHOLDERS or word in arguments.OPERATORS or flag_name(word) in public_skeleton_words:
            shipped.append(word)
        else:
            shipped.append(stand_ins.skeleton_word_stand_in(word, position in places, public_stand_ins, tool_tables))
    return " ".join(shipped)


def flag_name(word: str) -> str:
    """A skeleton word without a flag's `=<value>`."""
    return word[:-len(stand_ins.VALUE_SUFFIX)] if word.endswith(stand_ins.VALUE_SUFFIX) else word


def public_skeleton(skeleton: str, public_keys: frozenset[str], public_skeleton_words: frozenset[str],
                    public_stand_ins: frozenset[str], tool_tables: categories.ToolTables) -> str:
    """The skeleton with every key not on the public list replaced by `<key>`, and every hidden command word by
    its stand-in."""
    if skeleton == arguments.UNPARSED:
        return skeleton
    pairs = []
    for key, value in arguments.skeleton_pairs(skeleton):
        public_key = key if key in public_keys else KEY_MARK
        if value in arguments.VALUE_PLACEHOLDERS:
            public_value = value
        else:
            public_value = public_command(value, public_skeleton_words, public_stand_ins, tool_tables)
        pairs.append((public_key, public_value))
    return arguments.pairs_text(pairs)


def is_public_command(command_skeleton: str, public_skeleton_words: frozenset[str], public_stand_ins: frozenset[str]) -> bool:
    """Whether every word of a command skeleton is a placeholder, an operator, public, or a stand-in."""
    for word in command_skeleton.split():
        if not (word in arguments.PLACEHOLDERS or word in arguments.OPERATORS or flag_name(word) in public_skeleton_words
                or stand_ins.is_allowed_mark(word, public_stand_ins)):
            return False
    return True


def is_public_skeleton(skeleton: str, public_keys: frozenset[str], public_skeleton_words: frozenset[str],
                       public_stand_ins: frozenset[str], public_types: frozenset[str] = frozenset()) -> bool:
    """Whether a skeleton holds only public keys or `<key>`, and values that are placeholders, public commands, or a
    public `subagent_type`."""
    if skeleton == arguments.UNPARSED:
        return True
    try:
        pairs = arguments.skeleton_pairs(skeleton)
    except json.JSONDecodeError:
        return False
    if not isinstance(pairs, list):
        return False
    for key, value in pairs:
        if key != KEY_MARK and key not in public_keys:
            return False
        if not isinstance(value, str):
            return False
        if key == arguments.SUBAGENT_TYPE_KEY and value in public_types:
            continue
        if value not in arguments.VALUE_PLACEHOLDERS and not is_public_command(value, public_skeleton_words, public_stand_ins):
            return False
    return True


def anonymised_columns(calls: pyarrow.Table, words: public_words.PublicWords,
                       tool_tables: categories.ToolTables) -> dict[str, pyarrow.Array]:
    """The anonymised `name`, `programs` and `arguments_skeleton` of each call, from its full columns."""
    names = [public_words.released_name(raw_name, words) for raw_name in calls["raw_name"].to_pylist()]
    programs = []
    for full_labels in calls["programs_full"].to_pylist():
        if full_labels is None:
            programs.append(None)
        else:
            programs.append([anonymised_label(label, words, tool_tables) for label in full_labels])
    skeletons = []
    for skeleton, arguments_text in zip(calls["arguments_skeleton_full"].to_pylist(), calls["arguments"].to_pylist()):
        public = public_skeleton(skeleton, words.argument_keys, words.skeleton_words, words.stand_ins, tool_tables)
        skeletons.append(with_subagent_type(public, arguments.subagent_type(arguments_text), words.subagent_types))
    return {
        "name": pyarrow.array(names, pyarrow.string()),
        "programs": pyarrow.array(programs, pyarrow.list_(pyarrow.string())),
        "arguments_skeleton": pyarrow.array(skeletons, pyarrow.large_string()),
    }
