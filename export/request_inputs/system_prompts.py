"""The top-level system prompt of each kept request, from one read of its weekly log.

Claude Code sends its system prompt in the request payload's top-level
`system` field, which the record's `prompt` does not hold. This step reads the
weekly log once, in byte ranges, and keeps that field for every request of the
week's extract that has one, together with the fields of Claude Code's billing
header (`x-anthropic-billing-header: cc_version=…; cc_entrypoint=…;`), which a
later exclusion may be decided on. It writes `system_prompts.parquet` beside
the extract.
"""

import pathlib

import orjson
import pyarrow
from pyarrow import parquet

from export import parallel, paths
from export.log_records import fields


BILLING_HEADER_PREFIX = "x-anthropic-billing-header:"

SYSTEM_PROMPTS_SCHEMA = pyarrow.schema(
    [
        ("request_id", pyarrow.string()),
        ("system", pyarrow.large_string()),  # the payload's `system` value, as JSON text
        ("billing_header", pyarrow.map_(pyarrow.string(), pyarrow.string())),  # null without a header
    ]
)


def first_text(system: object) -> str:
    """The text the system prompt starts with: the string itself, or its first text block's."""
    if isinstance(system, str):
        return system
    if isinstance(system, list) and system and isinstance(system[0], dict):
        return fields.read_string(system[0].get("text")) or ""
    return ""


def billing_header_fields(system: object) -> dict[str, str] | None:
    """The `key=value` fields of a leading billing header, or None when the system prompt has none."""
    text = first_text(system)
    if not text.startswith(BILLING_HEADER_PREFIX):
        return None
    header_line = text[len(BILLING_HEADER_PREFIX):].split("\n", 1)[0]
    header = {}
    for part in header_line.split(";"):
        key, equals, value = part.partition("=")
        if equals:
            header[key.strip()] = value.strip()
    return header


def system_prompt_row(line: bytes, wanted: frozenset[str]) -> dict | None:
    """The row of one raw line: when its request is wanted and its payload has a top-level system prompt."""
    record = fields.parse_line(line)
    if record is None or fields.read_string(record.get("request_id")) not in wanted:
        return None
    payload_text = fields.read_body(record.get("request_payload"))
    payload = fields.parse_object(payload_text) if payload_text is not None else None
    if payload is None or not payload.get("system"):
        return None
    system = payload["system"]
    return {
        "request_id": record["request_id"],
        "system": orjson.dumps(system).decode(),
        "billing_header": billing_header_fields(system),
    }


def rows_in_range(week_file: pathlib.Path, start: int, end: int, wanted: frozenset[str]) -> list[dict]:
    """The rows of one byte range of the weekly log, in file order."""
    rows = []
    with open(week_file, "rb") as handle:
        for line in parallel.lines_in_range(handle, start, end):
            row = system_prompt_row(line, wanted)
            if row is not None:
                rows.append(row)
    return rows


def write_week_system_prompts(week: paths.Week) -> int:
    """Write the week's `system_prompts.parquet`, and return how many requests have a system prompt."""
    wanted = frozenset(parquet.read_table(week.folder / paths.EXTRACT_FILE, columns=["request_id"])["request_id"].to_pylist())
    rows = []
    with parallel.process_pool(week.processes) as pool:
        jobs = [pool.submit(rows_in_range, week.log_file, start, end, wanted)
                for start, end in parallel.split_into_ranges(week.log_file, week.processes)]
        for job in jobs:
            rows.extend(job.result())
    parallel.write_rows(rows, SYSTEM_PROMPTS_SCHEMA, week.folder / paths.SYSTEM_PROMPTS_FILE)
    return len(rows)
