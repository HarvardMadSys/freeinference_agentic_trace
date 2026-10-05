"""A call's arguments: parsed, with every value erased into a skeleton, and a shell command's words typed.

The skeleton is written as a JSON object's text: `{"file_path": "<path>",
"limit": "<num>"}`, every key kept and every value replaced by its type. A
shell call's command becomes its command skeleton: wrappers, labels, flags,
redirects and operators stay, and every other word becomes its type. A flag
whose name still holds an IP address once its value is erased is a value, so
that no IP address can reach a skeleton. Two keys may later both read `<key>`,
so a skeleton is always read back as a list of (key, value) pairs, never as a
dict.
"""

import dataclasses
import json
import re

import orjson

from export.tool_calls import commands, shell

PATH_MARK = "<path>"
NUMBER_MARK = "<num>"
STRING_MARK = "<str>"
VARIABLE_MARK = "<var>"
ASSIGNMENT_MARK = "<assign>"
VALUE_MARK = "<value>"
HEREDOC_MARK = "<heredoc>"
CODE_MARK = "<code>"
LINE_BREAK_MARK = ";"  # a line break between two commands is written as `;`
BACKGROUND_MARK = "&"
PLACEHOLDERS = frozenset({PATH_MARK, NUMBER_MARK, STRING_MARK, VARIABLE_MARK, ASSIGNMENT_MARK, HEREDOC_MARK, CODE_MARK})
OPERATORS = frozenset({"&&", "||", ";", "|", "&"})
PATH_SHAPED = re.compile(r"[/\\]|\.[A-Za-z0-9_]{1,8}$|[*?]")  # a separator, an extension, or a glob
BARE_PATHS = frozenset({".", "..", "~", "./", "../"})
VARIABLE = re.compile(r"^[$][\w{(]|^%\w+%")  # `$F`, `${path}`, `$(cmd)`, `%TEMP%`
NUMBER = re.compile(r"^\d+(\.\d+)?$")
REDIRECT = re.compile(r"^\d*(>>|>\||[<>])(&\d*)?")  # `>`, `>>`, `2>&1`, `<`
# A flag's name, kept in the skeleton: `-n`, `-5`, `--color`. A word that starts with a dash but is text,
# such as `---DONE---` or `-m"fix the bug"`, is a value, and is erased.
FLAG_NAME = re.compile(r"^--?[A-Za-z0-9][A-Za-z0-9_.:+-]*$")
# Programs whose short flags take their value glued on, so that every character after a short flag's letter is a
# value: `mysql -proot`, `sqlcmd -Psecret`, `sshpass -psecret`, `smbclient -Uuser%secret`. They are often run by
# another program (`docker exec db mysql -proot`), so every short flag after one of them is read this way.
GLUED_VALUE_PROGRAMS = frozenset({
    "mysql", "mysqldump", "mysqladmin", "mysqlimport", "mysqlshow", "mysqlcheck", "mariadb", "mariadb-dump",
    "mariadb-admin", "sshpass", "sqlcmd", "bcp", "isql", "sqlplus", "mongo", "mongosh", "mongodump", "mongorestore",
    "smbclient", "htpasswd", "redis-cli", "psql", "pg_dump", "pg_restore", "pg_dumpall",
})
# Four numbers joined by dots (`10.0.0.1`), not inside a longer run of digits and dots.
IPV4_ADDRESS = re.compile(r"(?<![\d.])\d{1,3}(\.\d{1,3}){3}(?![\d.])")
# A run of hexadecimal digits and colons, which is an IPv6 address when it holds two colons and a digit (`::1`).
IPV6_CANDIDATE = re.compile(r"[0-9A-Fa-f:]{3,}")
MIN_IPV6_COLONS = 2
SUCCESS = "success"
PARTIAL = "partial"
FAILED = "failed"
UNPARSED = "<unparsed>"  # arguments that are not a JSON object
BOOLEAN_MARK = "<bool>"
NULL_MARK = "<null>"
LIST_MARK = "<list>"
OBJECT_MARK = "<object>"
VALUE_PLACEHOLDERS = frozenset({PATH_MARK, NUMBER_MARK, STRING_MARK, BOOLEAN_MARK,
                                NULL_MARK, LIST_MARK, OBJECT_MARK, CODE_MARK, UNPARSED})
SUBAGENT_TYPE_KEY = "subagent_type"  # the argument of a spawn call naming the kind of subagent; its value may ship


@dataclasses.dataclass(frozen=True)
class ShellFacts:
    """What is kept of one shell call: its program labels, their operators, and its skeleton."""

    programs: list[str]
    joined_by: list[str]
    parse_status: str
    skeleton: str
    backgrounded: bool  # some command of it runs in the background
    command_field: str | None  # the argument that held the command or the code


NOT_A_COMMAND = ShellFacts([], [], FAILED, "", False, None)


def parsed_arguments(arguments_text: str | None) -> dict | None:
    """The arguments as a JSON object; None when they are missing or are not one."""
    if arguments_text is None:
        return None
    try:
        arguments = orjson.loads(arguments_text)
    except orjson.JSONDecodeError:
        return None
    return arguments if isinstance(arguments, dict) else None


def value_type(value: object) -> str:
    """The placeholder of one argument value."""
    if isinstance(value, bool):
        return BOOLEAN_MARK
    if isinstance(value, (int, float)):
        return NUMBER_MARK
    if value is None:
        return NULL_MARK
    if isinstance(value, list):
        return LIST_MARK
    if isinstance(value, dict):
        return OBJECT_MARK
    if NUMBER.match(value):
        return NUMBER_MARK
    return PATH_MARK if is_path(value) else STRING_MARK


def pairs_text(pairs: list[tuple[str, str]]) -> str:
    """(key, value) pairs written as a JSON object's text, repeated keys allowed."""
    members = [orjson.dumps(key).decode() + ": " + orjson.dumps(value).decode() for key, value in pairs]
    return "{" + ", ".join(members) + "}"


def full_skeleton(arguments: dict | None, shell_facts: ShellFacts | None) -> str:
    """The skeleton of a call's arguments; a shell call's command replaced by its command skeleton."""
    if arguments is None:
        return UNPARSED
    pairs = []
    for key, value in arguments.items():
        if shell_facts is not None and key == shell_facts.command_field and shell_facts.skeleton:
            pairs.append((str(key), shell_facts.skeleton))
        else:
            pairs.append((str(key), value_type(value)))
    return pairs_text(pairs)


def subagent_type(arguments_text: str | None) -> str | None:
    """The string value of a call's `subagent_type` argument; None without one."""
    parsed = parsed_arguments(arguments_text)
    value = parsed.get(SUBAGENT_TYPE_KEY) if parsed is not None else None
    return value if isinstance(value, str) else None


def skeleton_pairs(skeleton: str) -> list[tuple[str, str]]:
    """A skeleton read back as its (key, value) pairs, in order; no pairs for `<unparsed>`."""
    if skeleton == UNPARSED:
        return []
    return json.loads(skeleton, object_pairs_hook=list)


def command_values(skeleton: str) -> list[str]:
    """The command skeletons among a skeleton's values; a spawn call's subagent type is a name, not a command."""
    return [value for key, value in skeleton_pairs(skeleton)
            if value not in VALUE_PLACEHOLDERS and key != SUBAGENT_TYPE_KEY]


def shell_facts(arguments: dict | None) -> ShellFacts:
    """The facts of one shell call, from its parsed arguments."""
    if arguments is None:
        return NOT_A_COMMAND
    field, text = commands.command_source(arguments)
    if field == commands.CODE_FIELD:
        label = f"{commands.code_language(arguments)} {commands.CODE_LABEL_SUFFIX}"
        return ShellFacts([label], [""], SUCCESS, CODE_MARK, False, field)
    if text is None:
        return NOT_A_COMMAND
    command = shell.parse(text)
    if not command.invocations:
        return dataclasses.replace(NOT_A_COMMAND, command_field=field)
    return ShellFacts(
        programs=[invocation.label for invocation in command.invocations],
        joined_by=[invocation.joined_by for invocation in command.invocations],
        parse_status=PARTIAL if command.has_syntax_error else SUCCESS,
        skeleton=skeleton_of(command.invocations),
        backgrounded=any(invocation.backgrounded for invocation in command.invocations),
        command_field=field,
    )


def skeleton_of(invocations: list[shell.Invocation]) -> str:
    """The invocations, each typed, joined by the operators written between them."""
    pieces = []
    for position, invocation in enumerate(invocations):
        if position > 0:
            previous = invocations[position - 1]
            pieces.append(BACKGROUND_MARK if previous.backgrounded else invocation.joined_by or LINE_BREAK_MARK)
        pieces.append(invocation_skeleton(invocation))
    if invocations and invocations[-1].backgrounded:
        pieces.append(BACKGROUND_MARK)
    return " ".join(pieces)


def invocation_skeleton(invocation: shell.Invocation) -> str:
    """A wrapper's name and the label stay as written; every other word becomes its type."""
    words = [prefix_word(word) for word in invocation.prefix] + [invocation.label]
    glued_values = any(names_glued_value_program(word) for word in invocation.label.split()[:1])
    for word in invocation.arguments:
        glued_values = glued_values or names_glued_value_program(word)
        words.append(type_word(word, glued_values))
    if invocation.opened_heredoc:
        words.append(HEREDOC_MARK)
    return " ".join(words)


def prefix_word(word: str) -> str:
    """A word before the program: a wrapper stays, an assignment or a wrapper's value becomes its type."""
    if shell.ASSIGNMENT.match(word):
        return ASSIGNMENT_MARK
    if shell.program_basename(word) in shell.WRAPPERS:
        return shell.program_basename(word)
    return type_word(word)


def names_glued_value_program(word: str) -> bool:
    """Whether a word runs a program whose short flags take their value glued on."""
    return shell.program_basename(word).lower() in GLUED_VALUE_PROGRAMS


def type_word(word: str, glued_values: bool = False) -> str:
    """One argument word with its value erased: a flag's name and a redirect stay, a value becomes its type. With
    `glued_values`, every character after a short flag's letter is a value. A flag whose name still holds an IP
    address once its value is erased is a value."""
    if not word:
        return STRING_MARK
    if shell.ASSIGNMENT.match(word):
        return ASSIGNMENT_MARK
    if VARIABLE.match(word):
        return VARIABLE_MARK
    if shell.is_flag(word):
        flag = flag_skeleton(word, glued_values)
        return STRING_MARK if holds_ip_address(flag) else flag
    redirect = REDIRECT.match(word)
    if redirect:
        return redirect.group(0) + (" " + PATH_MARK if word[redirect.end():] else "")
    if NUMBER.match(word):
        return NUMBER_MARK
    if is_path(word):
        return PATH_MARK
    return STRING_MARK


def flag_skeleton(word: str, glued_values: bool) -> str:
    """A flag with its value erased: `--port=<value>`. A short flag carries its value glued on (`-proot`,
    `-h127.0.0.1`, `-A10`) with `glued_values`, or when anything but letters follows its letter;
    then only its letter stays, written `-p=<value>`. A count such as `-20` stays. A flag whose name is text is a
    value."""
    name, has_value = word.split("=", 1)[0], "=" in word
    if not FLAG_NAME.match(name):
        return STRING_MARK
    glued = name[2:]
    is_lettered_short = not name.startswith("--") and name[1].isalpha()  # `-20` is a count, not a flag with a value
    if is_lettered_short and glued and (glued_values or not glued.isalpha()):
        name, has_value = name[:2], True
    return name + "=" + VALUE_MARK if has_value else name


def holds_ip_address(text: str) -> bool:
    """Whether the text holds an IPv4 address, or an IPv6 one (two colons and a digit among hexadecimal digits)."""
    if IPV4_ADDRESS.search(text):
        return True
    for candidate in IPV6_CANDIDATE.findall(text):
        if candidate.count(":") >= MIN_IPV6_COLONS and any(character.isdigit() for character in candidate):
            return True
    return False


def is_path(word: str) -> bool:
    """Whether a word looks like a path: it has a separator or an extension, is a glob, or is `.`, `..` or `~`."""
    return word in BARE_PATHS or PATH_SHAPED.search(word) is not None


def skeleton_words(skeleton: str) -> list[str]:
    """The words of a command skeleton that are neither placeholders nor operators, a flag's `=<value>` removed."""
    words = []
    for word in skeleton.split():
        if word in PLACEHOLDERS or word in OPERATORS:
            continue
        words.append(word[:-len("=" + VALUE_MARK)] if word.endswith("=" + VALUE_MARK) else word)
    return words
