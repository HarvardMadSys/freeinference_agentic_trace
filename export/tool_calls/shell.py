"""Reading a shell command: which programs ran, in order, and how each was joined to the one before.

The command is parsed as bash by tree-sitter and the syntax tree is walked, so
an operator inside quotes or a heredoc is never mistaken for one between
commands. The walk follows these rules:
- a comment, a heredoc body and a `$(...)` substitution are not read: in
  `make -j$(nproc)` only `make` runs;
- the commands inside a loop or a group run, so they are read;
- wrappers (`sudo -u www`, `timeout 60`, `nohup`) and assignments (`FOO=1`)
  are skipped, and the command they wrap is the one that ran;
- a first word that is source code (`import`, `def`) is not a program: the
  agent pasted a script where a command was expected.
"""

import dataclasses
import re

import tree_sitter
import tree_sitter_bash

PARSER = tree_sitter.Parser(tree_sitter.Language(tree_sitter_bash.language()))

WRAPPERS = frozenset({"sudo", "env", "time", "nohup", "exec", "command", "builtin", "nice", "ionice", "stdbuf",
                      "timeout", "watch", "setsid", "caffeinate", "xargs", "rtk"})  # each runs the command after it
WRAPPER_VALUE_FLAGS = frozenset({"-u", "-g", "-p", "-C", "-h", "-w", "-n", "--user", "--group", "-I", "-S"})  # take the next word
SOURCE_CODE_WORDS = frozenset({"import", "from", "def", "class", "with", "except", "try", "finally", "raise",
                               "assert", "lambda", "global", "nonlocal", "del", "pass", "yield", "await", "async",
                               "const", "let", "var", "console", "require", "print", "self", "this"})
# Programs whose first plain argument is a subcommand worth keeping in the label: `git commit`.
SUBCOMMAND_PROGRAMS = frozenset({"git", "npm", "npx", "pnpm", "yarn", "pip", "cargo", "docker", "podman",
                                 "docker-compose", "kubectl", "helm", "go", "apt", "apt-get", "brew", "poetry", "uv",
                                 "conda", "gh", "systemctl", "make", "terraform", "gcloud", "aws", "rustup", "deno",
                                 "bun", "dotnet", "mvn", "gradle", "rails", "bundle", "composer", "gem", "dvc",
                                 "pre-commit", "pdm", "rye", "pipx", "hatch", "module", "svn", "hg"})
RUNNER_PROGRAMS = frozenset({"uv", "poetry", "pdm", "rye", "pipx", "hatch", "conda"})  # `uv run pytest` runs pytest
INTERPRETER_ALIASES = {"python3": "python", "python2": "python", "py": "python", "pip3": "pip", "node18": "node",
                       "nodejs": "node"}

ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
SUBCOMMAND_WORD = re.compile(r"^[A-Za-z][A-Za-z0-9:_-]*$")
PROGRAM_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")
DURATION = re.compile(r"^[0-9.]+[smh]?$")  # a wrapper's value: `timeout 60`, `sleep 1.5s`
VERSIONED_PYTHON = re.compile(r"^python[0-9.]+$")  # `python3.11` is `python`
FLAG = re.compile(r"^-.+")  # `-n`, `--flag`, `--flag=value`; a lone `-` is an operand

SKIPPED_NODES = frozenset({"comment", "command_substitution", "heredoc_body"})
OPERATOR_TOKENS = {"&&": "&&", "||": "||", ";": ";", "|": "|", "|&": "|"}
BACKGROUND_TOKEN = "&"
QUOTED_NODES = frozenset({"string", "raw_string"})
REDIRECT_NODES = frozenset({"file_redirect", "herestring_redirect"})
HEREDOC_NODE = "heredoc_redirect"


@dataclasses.dataclass
class Invocation:
    """One program that ran: the words before it, its label, its arguments, and the operator before it."""

    prefix: list[str]
    label: str
    arguments: list[str]
    joined_by: str
    backgrounded: bool = False
    opened_heredoc: bool = False


@dataclasses.dataclass(frozen=True)
class ShellCommand:
    """The programs a command ran, and whether some of its text did not parse."""

    invocations: list[Invocation]
    has_syntax_error: bool


def program_basename(word: str) -> str:
    """A program word without its folder or `.exe`, interpreter spellings folded: `/usr/bin/python3` is `python`."""
    name = re.split(r"[/\\]", word)[-1]
    if name.endswith(".exe"):
        name = name[:-len(".exe")]
    if VERSIONED_PYTHON.match(name):
        return "python"
    return INTERPRETER_ALIASES.get(name, name)


def is_flag(word: str) -> bool:
    """Whether a word is an option: `-n`, `--flag`, `--flag=value`."""
    return FLAG.match(word) is not None


def parse(command_text: str) -> ShellCommand:
    """The invocations of one command text, in the order they appear."""
    tree = PARSER.parse(command_text.encode())
    found: list[Invocation] = []
    walk(tree.root_node, "", found)
    return ShellCommand(found, tree.root_node.has_error)


def walk(node, joined_by: str, found: list[Invocation]) -> None:
    """Append every invocation under `node` to `found`; `joined_by` is the operator before the first."""
    if node.type in SKIPPED_NODES:
        return
    if node.type == "command":
        add_command(found, node, joined_by, [])
        return
    if node.type == "redirected_statement":
        body, redirects = node.children[0], node.children[1:]
        if body.type == "command":
            add_command(found, body, joined_by, redirects)
        else:
            add_to_last(found, body, joined_by, redirects)
        return
    operator = joined_by
    for child in node.children:
        if child.type in OPERATOR_TOKENS:
            operator = OPERATOR_TOKENS[child.type]
        elif child.type == BACKGROUND_TOKEN:
            if found:
                found[-1].backgrounded = True
            operator = ""
        elif child.is_named:
            count_before = len(found)
            walk(child, operator, found)
            if len(found) > count_before:
                operator = ""  # the next command starts a new line unless an operator says otherwise


def add_command(found: list[Invocation], command_node, joined_by: str, redirects: list) -> None:
    """Read one command node, with the redirects written after it, into `found`."""
    words = []
    for child in command_node.children:
        if child.type in REDIRECT_NODES:
            words.extend(redirect_words(child))
        elif child.is_named:
            words.append(word_text(child))
    extra_words, opened_heredoc, after_marker = read_redirects(redirects)
    invocation = make_invocation(words + extra_words, joined_by)
    if invocation is not None:
        invocation.opened_heredoc = opened_heredoc
        found.append(invocation)
    for statement in after_marker:
        walk(statement, "", found)


def add_to_last(found: list[Invocation], body, joined_by: str, redirects: list) -> None:
    """A redirect after a pipeline or list (`a | b > f`) belongs to its last command."""
    count_before = len(found)
    walk(body, joined_by, found)
    extra_words, opened_heredoc, after_marker = read_redirects(redirects)
    if len(found) > count_before:
        found[-1].arguments.extend(extra_words)
        found[-1].opened_heredoc = opened_heredoc
    for statement in after_marker:
        walk(statement, "", found)


def read_redirects(redirects: list) -> tuple[list[str], bool, list]:
    """The redirects' words, whether one opened a heredoc, and the statements written after a heredoc's marker.

    The grammar files what follows a heredoc marker on its line (`> out.py`,
    `| python3 -`) under the heredoc node itself.
    """
    words, after_marker = [], []
    for redirect in redirects:
        if redirect.type in REDIRECT_NODES:
            words.extend(redirect_words(redirect))
        elif redirect.type == HEREDOC_NODE:
            for part in redirect.children:
                if part.type in REDIRECT_NODES:
                    words.extend(redirect_words(part))
                elif part.is_named:
                    after_marker.append(part)
    opened_heredoc = any(redirect.type == HEREDOC_NODE for redirect in redirects)
    return words, opened_heredoc, after_marker


def word_text(node) -> str:
    """A node's text as a shell word: quotes removed, a command name unwrapped."""
    if node.type == "command_name" and node.children:
        node = node.children[0]
    text = node.text.decode(errors="replace")
    if node.type in QUOTED_NODES and len(text) >= 2:
        return text[1:-1]
    return text


def redirect_words(node) -> list[str]:
    """`2>&1` is one word; `> out.log` is two, the redirect and its target."""
    parts = [word_text(child) for child in node.children]
    if len(parts) >= 2 and parts[-2] in (">&", "<&"):
        return ["".join(parts)]
    return ["".join(parts[:-1]), parts[-1]]


def make_invocation(words: list[str], joined_by: str) -> Invocation | None:
    """The invocation these words describe; None when they run no program."""
    prefix, rest = split_prefix(words)
    if not rest or rest[0] in SOURCE_CODE_WORDS or rest[1:2] == ["="]:
        return None
    program = program_basename(rest[0])
    if not PROGRAM_NAME.match(program):
        return None
    label, arguments = make_label(program, rest[1:])
    return Invocation(prefix, label, arguments, joined_by)


def split_prefix(words: list[str]) -> tuple[list[str], list[str]]:
    """The assignments and wrappers before the program, with their options; then the program onwards."""
    index = 0
    while index < len(words):
        word = words[index]
        if ASSIGNMENT.match(word):
            index += 1
            continue
        if program_basename(word) not in WRAPPERS:
            break
        index += 1
        while index < len(words):
            option = words[index]
            if option in WRAPPER_VALUE_FLAGS:
                index += 2
            elif is_flag(option) or DURATION.match(option) or ASSIGNMENT.match(option):
                index += 1
            else:
                break
    return words[:index], words[index:]


def make_label(program: str, rest: list[str]) -> tuple[str, list[str]]:
    """The shortest string that says what ran (`git commit`, `python -m pytest`, `uv run pytest`), and the rest."""
    if program == "python":
        if rest[:1] == ["-m"] and len(rest) > 1:
            return f"python -m {rest[1]}", rest[2:]
        if rest[:1] == ["-c"]:
            return "python -c", rest[1:]
        return program, rest
    if program not in SUBCOMMAND_PROGRAMS:
        return program, rest
    subcommand = subcommand_index(rest, 0)
    if subcommand is None:
        return program, rest
    label, used = f"{program} {rest[subcommand]}", {subcommand}
    if program in RUNNER_PROGRAMS and rest[subcommand] == "run":
        inner = subcommand_index(rest, subcommand + 1)
        if inner is not None:
            label, used = f"{label} {rest[inner]}", used | {inner}
    return label, [word for index, word in enumerate(rest) if index not in used]


def subcommand_index(words: list[str], start: int) -> int | None:
    """The first word at or after `start` that could be a subcommand: not a flag, a path or a key=value."""
    for index in range(start, len(words)):
        if SUBCOMMAND_WORD.match(words[index]):
            return index
    return None
