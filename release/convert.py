"""A release week's trace lines converted into the three tables the analysis reads.

It checks each `traces/<day>.jsonl` against the week's manifest, converts the
lines (`trace_lines.read_tables`), and writes `sessions.parquet`,
`requests.parquet` and `tool_calls.parquet` into the week's folder, with
`tables.json`: the sha256 of the manifest they came from, and each table's
bytes, sha256 and rows. Each file replaces the old one only once it is
complete, by a rename, so a hard link to the old file (a download keeps one in
`data/hub`) is left as it was. The weeks converted are the listed ones, or
every week whose trace files are all downloaded.
"""

import dataclasses
import json
import pathlib

import pyarrow
from pyarrow import parquet

from release import reader
from release import schema, trace_lines

# Rows per row group: a reader takes a table one row group at a time, and ten
# thousand requests' block ids are about a hundred million values.
ROW_GROUP_ROWS = 10_000


class ConvertError(Exception):
    """A week cannot be converted: it has none or only some of its trace files, or no week has them."""


@dataclasses.dataclass(frozen=True)
class WeeksToConvert:
    """The week folders to convert, oldest first, and those left out because none of their trace files is downloaded."""

    with_traces: list[pathlib.Path]
    without_traces: list[pathlib.Path]


def write_table(week_directory: pathlib.Path, file_name: str, table: pyarrow.Table) -> dict:
    """Write one table, replacing it only once it is complete; return its bytes, sha256 and rows."""
    partial = week_directory / (file_name + ".partial")
    parquet.write_table(table, partial, compression="zstd", row_group_size=ROW_GROUP_ROWS)
    partial.rename(week_directory / file_name)
    return {"bytes": (week_directory / file_name).stat().st_size,
            "sha256": reader.file_sha256(week_directory / file_name), "rows": table.num_rows}


def convert_week(week_directory: pathlib.Path) -> dict:
    """Convert a released week's trace lines into its three tables and `tables.json`; return `tables.json`."""
    day_files = reader.checked_day_files(week_directory)
    tables = trace_lines.read_tables(day_files)
    files = {
        schema.SESSIONS_FILE: write_table(week_directory, schema.SESSIONS_FILE, tables.sessions),
        schema.REQUESTS_FILE: write_table(week_directory, schema.REQUESTS_FILE, tables.requests),
        schema.TOOL_CALLS_FILE: write_table(week_directory, schema.TOOL_CALLS_FILE, tables.tool_calls),
    }
    record = {"manifest_sha256": reader.file_sha256(week_directory / schema.MANIFEST_FILE), "files": files}
    partial = week_directory / (schema.TABLES_FILE + ".partial")
    partial.write_text(json.dumps(record, indent=2) + "\n")
    partial.rename(week_directory / schema.TABLES_FILE)
    return record


def weeks_to_convert(release_directory: pathlib.Path, weeks: list[str] | None) -> WeeksToConvert:
    """The listed weeks, each with all its trace files; or, with none listed, every week that has all of them.

    A week with only some of its trace files stops the conversion, as does a
    listed week with none; unlisted weeks with none are left out.
    """
    with_traces = []
    without_traces = []
    for week_directory in reader.find_weeks(release_directory, weeks):
        names = reader.checked_manifest(week_directory)["files"]
        present = sum(1 for name in names if (week_directory / name).exists())
        expected = len(names)
        if present == expected:
            with_traces.append(week_directory)
        elif present == 0 and weeks is None:
            without_traces.append(week_directory)
        else:
            raise ConvertError(f"{week_directory.name} has {present} of its {expected} trace files; run "
                               f"`python -m release download --traces --week {reader.week_sunday(week_directory)}`")
    if not with_traces:
        raise ConvertError(f"no week in {release_directory} has its trace files; run `python -m release download --traces`")
    return WeeksToConvert(with_traces, without_traces)
