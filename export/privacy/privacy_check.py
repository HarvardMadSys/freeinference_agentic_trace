"""The check before a release week is written: nothing private may ship.

A week is stopped, and the offending column named, when a file has a column
the release format does not list, when a raw id of the week appears in a
string column or a list of strings, when any string value holds an IP
address, or when a string value, or an element of a list of strings, is not
of its column's allowed form. A tool name, program
label, argument key or command word is allowed only when it is on the public
list of tool words, or is its anonymised stand-in.
"""

import re

import pyarrow

from release import schema
from export.privacy import public_words, stand_ins, tool_columns
from export.tool_calls import arguments

MIN_CHECKED_ID_CHARACTERS = 12  # shorter raw ids, such as `call_0`, are too common to check
MAX_MODEL_CHARACTERS = 200  # a model id is short, printable text

HEX = re.compile(r"[0-9a-f]+")
PRINTABLE = re.compile(r"[ -~]+")
PROVIDER_LETTERS = re.compile(r"[A-Z]{1,2}")  # a provider's anonymous letter, never its name


class PrivateValue(Exception):
    """A release file holds something that must not ship."""


def check_columns(table: pyarrow.Table, table_schema: pyarrow.Schema, file_name: str) -> None:
    """Stop unless the table has exactly the columns the release format lists."""
    if table.schema.names != table_schema.names:
        extra = sorted(set(table.schema.names) - set(table_schema.names))
        missing = sorted(set(table_schema.names) - set(table.schema.names))
        raise PrivateValue(f"{file_name}: columns differ from the release format: extra {extra}, missing {missing}")


def string_values(table: pyarrow.Table, name: str) -> set | None:
    """The distinct values of a string column, or the distinct elements of a list-of-strings column; None otherwise."""
    column_type = table.schema.field(name).type
    if pyarrow.types.is_string(column_type):
        return set(table.column(name).to_pylist())
    if pyarrow.types.is_list(column_type) and pyarrow.types.is_string(column_type.value_type):
        return set(table.column(name).combine_chunks().flatten().to_pylist())
    return None


def check_no_raw_ids(table: pyarrow.Table, raw_ids: set[str], file_name: str) -> None:
    """Stop when any raw id of 12 or more characters is a value of a string column or a list of strings."""
    long_ids = {raw_id for raw_id in raw_ids if raw_id and len(raw_id) >= MIN_CHECKED_ID_CHARACTERS}
    for name in table.schema.names:
        values = string_values(table, name)
        if values is None:
            continue
        leaked = values & long_ids
        if leaked:
            raise PrivateValue(f"{file_name}: column {name} holds a raw id, such as {sorted(leaked)[0]}")


# The tool-call columns whose values come from a fixed list.
FIXED_VALUES = {
    "category": schema.TOOL_CATEGORIES,
    "parse_status": schema.PARSE_STATUSES,
    "outcome": schema.OUTCOMES,
}


def allowed_tool_form(column: str, value: str, words: public_words.PublicWords) -> bool:
    """Whether a value of a tool-call column is allowed: a fixed value, a public word, or an anonymised stand-in."""
    if column in FIXED_VALUES:
        return value in FIXED_VALUES[column]
    if column in ("name", "tool_definition_names"):
        return value == schema.OTHER_WORD or value in words.names
    if column == "programs":
        return stand_ins.is_allowed_label(value, words.programs, words.stand_ins)
    if column == "arguments_skeleton":
        return tool_columns.is_public_skeleton(value, words.argument_keys, words.skeleton_words, words.stand_ins,
                                            words.subagent_types)
    return False


def allowed_form(column: str, value: str, harnesses: frozenset[str], words: public_words.PublicWords) -> bool:
    """Whether a string value has its column's allowed form."""
    if column in ("session_id", "parent_session_id"):
        return len(value) == schema.SESSION_ID_HEX_CHARACTERS and HEX.fullmatch(value) is not None
    if column == "user_id":
        return len(value) == schema.USER_ID_HEX_CHARACTERS and HEX.fullmatch(value) is not None
    if column == "step_trigger":
        return value in schema.STEP_TRIGGERS
    if column == "waited_for":
        return value in schema.WAITED_FOR
    if column == "harness":
        return value in harnesses
    if column == "model":
        return len(value) <= MAX_MODEL_CHARACTERS and PRINTABLE.fullmatch(value) is not None
    if column == "provider":
        return PROVIDER_LETTERS.fullmatch(value) is not None
    if column == "link_reason":
        return value in schema.LINK_REASONS
    if column == "finish_reason":
        return value in schema.FINISH_REASONS
    if column == "transition":
        return value in schema.TRANSITIONS
    if column == "cause":
        return value in schema.CAUSES
    return allowed_tool_form(column, value, words)


def check_forms(table: pyarrow.Table, harnesses: frozenset[str], words: public_words.PublicWords,
                file_name: str) -> None:
    """Stop when a string value, or an element of a list of strings, is not of its column's allowed form."""
    for name in table.schema.names:
        values = string_values(table, name)
        if values is None:
            continue
        for value in values:
            if value is not None and not allowed_form(name, value, harnesses, words):
                raise PrivateValue(f"{file_name}: column {name} holds a value of the wrong form: {value[:40]!r}")


def check_no_ip_addresses(table: pyarrow.Table, file_name: str) -> None:
    """Stop when any string value, or element of a list of strings, holds an IP address; the address is not shown."""
    for name in table.schema.names:
        values = string_values(table, name)
        if values is None:
            continue
        if any(value is not None and arguments.holds_ip_address(value) for value in values):
            raise PrivateValue(f"{file_name}: column {name} holds an IP address")


def check_week(tables: dict[str, pyarrow.Table], raw_ids: set[str], harnesses: frozenset[str],
               words: public_words.PublicWords = public_words.NO_WORDS) -> None:
    """Every check, on every file of a week, before anything is written."""
    for file_name, table in tables.items():
        check_columns(table, schema.SCHEMAS[file_name], file_name)
        check_no_raw_ids(table, raw_ids, file_name)
        check_no_ip_addresses(table, file_name)
        check_forms(table, harnesses, words, file_name)
