"""Where a shell call's command sits in its arguments.

Harnesses put the command in different places: `command`, `commands` (a list,
run in order) or `cmd`. A call with `code` and a `language` asked for code to
be run; it has no command line, and is one program, `<language> (code)`.
"""

COMMAND_FIELDS = ("command", "commands", "cmd")  # tried in this order; the first that holds a command is read
LIST_FIELD = "commands"  # the one that may hold a list of commands, run in order
CODE_FIELD = "code"  # code to run instead of a command, when there is no `command` or `commands`
LANGUAGE_FIELDS = ("language", "lang")  # the code's language, tried in order
DEFAULT_CODE_LANGUAGE = "code"
CODE_LABEL_SUFFIX = "(code)"


def command_source(arguments: dict) -> tuple[str | None, str | None]:
    """The argument that holds the command or the code, and the command's text; (None, None) when there is none."""
    if CODE_FIELD in arguments and "command" not in arguments and LIST_FIELD not in arguments:
        return CODE_FIELD, None
    for field in COMMAND_FIELDS:
        text = command_text(arguments.get(field), field == LIST_FIELD)
        if text is not None:
            return field, text
    return None, None


def command_text(value: object, may_be_list: bool) -> str | None:
    """A command argument as one text: a string as it is, a list of commands joined by line breaks."""
    if isinstance(value, str):
        return value
    if not may_be_list or not isinstance(value, list):
        return None
    commands = []
    for item in value:
        if isinstance(item, str):
            commands.append(item)
        elif isinstance(item, dict) and isinstance(item.get("command"), str):
            commands.append(item["command"])
    return "\n".join(commands) if commands else None


def code_language(arguments: dict) -> str:
    """The language of a code argument, lower-cased."""
    for field in LANGUAGE_FIELDS:
        if arguments.get(field):
            return str(arguments[field]).lower()
    return DEFAULT_CODE_LANGUAGE
