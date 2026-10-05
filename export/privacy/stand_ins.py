"""What a hidden tool word ships as: its public prefix, a typed stand-in, or a plain mark.

A program label that is not public keeps its longest public prefix, followed
by `<word>`: `mvn install` becomes `mvn <word>`. A label with no public prefix
gets one typed candidate: its purpose, from the purpose lists the
categorisation already uses (`<packages>`); or, when it has no purpose, its
script's extension (`<script>.sh`). The candidate ships only when enough
accounts produced it, which the public list's `stand_ins` records; otherwise,
and when there is no candidate, the label becomes `<program>`.

In a command skeleton, a hidden word in a program's place (the first word of
a command that is not a wrapper, a placeholder or a flag) gets the same
treatment, read as a label on its own. A hidden flag keeps its shape:
`--<flag>`, `-<flag>`, each with its `=<value>`. Any other hidden word is
`<word>`.
"""

import re

from export.tool_calls import arguments, categories, shell

WORD_MARK = "<word>"
PROGRAM_MARK = "<program>"
LONG_FLAG_MARK = "--<flag>"
SHORT_FLAG_MARK = "-<flag>"
VALUE_SUFFIX = "=" + arguments.VALUE_MARK
# The flag stand-ins, which name nothing and are always allowed.
FLAG_MARKS = frozenset({LONG_FLAG_MARK, SHORT_FLAG_MARK, LONG_FLAG_MARK + VALUE_SUFFIX, SHORT_FLAG_MARK + VALUE_SUFFIX})
# The extensions of the scripts an agent runs by name; a script with another extension becomes `<program>`.
SCRIPT_EXTENSIONS = ("sh", "bash", "py", "js", "mjs", "cjs", "ts", "ps1", "rb", "pl", "bat", "cmd", "php", "lua")
SCRIPT_NAME = re.compile(r"\.(" + "|".join(SCRIPT_EXTENSIONS) + r")$", re.IGNORECASE)


def typed_candidate(label: str, tool_tables: categories.ToolTables) -> str | None:
    """A hidden label's one typed stand-in: `<purpose>` when the purpose lists know it, else `<script>.<ext>` when
    its first word is a script; None when it is neither."""
    purpose = categories.purpose_of_label(label, tool_tables)
    if purpose != categories.OTHER_PURPOSE:
        return f"<{purpose}>"
    words = label.split()
    script = SCRIPT_NAME.search(words[0]) if words else None
    if script:
        return f"<script>.{script.group(1).lower()}"
    return None


def public_prefix(label: str, public_programs: frozenset[str]) -> str | None:
    """The longest leading run of the label's whole words that is a public label, shorter than the label."""
    words = label.split()
    for length in range(len(words) - 1, 0, -1):
        prefix = " ".join(words[:length])
        if prefix in public_programs:
            return prefix
    return None


def label_stand_in(label: str, public_programs: frozenset[str], public_stand_ins: frozenset[str],
                   tool_tables: categories.ToolTables) -> str:
    """What a hidden program label ships as."""
    prefix = public_prefix(label, public_programs)
    if prefix is not None:
        return f"{prefix} {WORD_MARK}"
    candidate = typed_candidate(label, tool_tables)
    if candidate is not None and candidate in public_stand_ins:
        return candidate
    return PROGRAM_MARK


def flag_stand_in(word: str) -> str | None:
    """A hidden flag's stand-in, keeping its dashes and its `=<value>`; None when the word is not a flag."""
    suffix = VALUE_SUFFIX if word.endswith(VALUE_SUFFIX) else ""
    if word.startswith("--"):
        return LONG_FLAG_MARK + suffix
    if word.startswith("-"):
        return SHORT_FLAG_MARK + suffix
    return None


def program_places(words: list[str]) -> set[int]:
    """The positions of a command skeleton's words that are in a program's place: in each command, the first word
    that is not a wrapper, a placeholder or a flag."""
    places = set()
    looking = True
    for position, word in enumerate(words):
        if word in arguments.OPERATORS:
            looking = True
        elif looking and not (word in shell.WRAPPERS or word in arguments.PLACEHOLDERS or word.startswith("-")):
            places.add(position)
            looking = False
    return places


def skeleton_word_stand_in(word: str, at_program_place: bool, public_stand_ins: frozenset[str],
                           tool_tables: categories.ToolTables) -> str:
    """What a hidden word of a command skeleton ships as."""
    if at_program_place:
        candidate = typed_candidate(word, tool_tables)
        return candidate if candidate is not None and candidate in public_stand_ins else PROGRAM_MARK
    flag = flag_stand_in(word)
    return flag if flag is not None else WORD_MARK


def is_allowed_label(value: str, public_programs: frozenset[str], public_stand_ins: frozenset[str]) -> bool:
    """Whether a released program label is public, a public prefix followed by `<word>`, or a stand-in."""
    if value in public_programs or value == PROGRAM_MARK or value in public_stand_ins:
        return True
    prefix, _, rest = value.rpartition(" ")
    return rest == WORD_MARK and prefix in public_programs


def is_allowed_mark(word: str, public_stand_ins: frozenset[str]) -> bool:
    """Whether a released skeleton word is a stand-in: `<word>`, `<program>`, a flag stand-in, or a public typed
    stand-in."""
    return word in (WORD_MARK, PROGRAM_MARK) or word in FLAG_MARKS or word in public_stand_ins
