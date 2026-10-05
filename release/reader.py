"""Loading weeks of the anonymous release into data frames.

The reader reads the tables converted from a week's trace lines
(`release/convert.py`). Each table is checked against the week's
`tables.json` before it is read; the trace files are checked when they are converted. `load`
gives the sessions, the requests and the tool calls; the requests leave out their block
ids, which `open_blocks` reads one row group at a time, and their role
tokens are flattened into `role_tokens.system`, `role_tokens.user`,
`role_tokens.assistant`, `role_tokens.tool` and `role_tokens.tool_definitions`.
"""

import dataclasses
import hashlib
import json
import pathlib

import pandas
from pyarrow import parquet

from release import schema

HASH_CHUNK_BYTES = 16 * 1024 * 1024  # read size when hashing a file
BLOCK_IDS = "block_ids"


class ReleaseError(Exception):
    """A release week is missing, corrupted, or of another format."""


@dataclasses.dataclass
class Release:
    """The sessions, requests and tool calls of the loaded weeks."""

    requests: pandas.DataFrame
    sessions: pandas.DataFrame
    tool_calls: pandas.DataFrame


def load(release_directory: str | pathlib.Path, weeks: list[str] | None = None) -> Release:
    """The sessions, requests and tool calls of the listed weeks, each given by its Sunday, or of every week."""
    frames = {file_name: [] for file_name in schema.SCHEMAS}
    for week_directory in week_directories(release_directory, weeks):
        for file_name in schema.SCHEMAS:
            frames[file_name].append(read_file(week_directory, file_name))
    return Release(
        requests=pandas.concat(frames[schema.REQUESTS_FILE], ignore_index=True),
        sessions=pandas.concat(frames[schema.SESSIONS_FILE], ignore_index=True),
        tool_calls=pandas.concat(frames[schema.TOOL_CALLS_FILE], ignore_index=True),
    )


def week_directories(release_directory: str | pathlib.Path, weeks: list[str] | None = None) -> list[pathlib.Path]:
    """The listed week folders of a release, or every one, each checked against its manifest."""
    chosen = find_weeks(pathlib.Path(release_directory).expanduser(), weeks)
    for week_directory in chosen:
        check_week(week_directory)
    return chosen


def open_blocks(week_directory: pathlib.Path) -> parquet.ParquetFile:
    """A checked week's `requests.parquet`, opened to read its `block_ids` one row group at a time.

    A week holds billions of block ids, too many for a data frame; the cache
    simulation reads them as Arrow arrays, in the file's session and step order.
    """
    return parquet.ParquetFile(week_directory / schema.REQUESTS_FILE)


def week_sunday(week_directory: pathlib.Path) -> str:
    """The Sunday a week folder is named by, as `YYYY-MM-DD`: the start of `<sunday>_<saturday>`."""
    return week_directory.name[:len("YYYY-MM-DD")]


def find_weeks(release_directory: pathlib.Path, weeks: list[str] | None) -> list[pathlib.Path]:
    """The week folders to read, in week order."""
    available = sorted(path for path in release_directory.iterdir() if (path / schema.MANIFEST_FILE).exists())
    if weeks is None:
        chosen = available
    else:
        chosen = [path for path in available if week_sunday(path) in weeks]
        missing = sorted(set(weeks) - {week_sunday(path) for path in chosen})
        if missing:
            raise ReleaseError(f"no release week starts on {', '.join(missing)} in {release_directory}")
    if not chosen:
        raise ReleaseError(f"{release_directory} holds no release week")
    return chosen


def checked_manifest(week_directory: pathlib.Path) -> dict:
    """The week's manifest; stop unless it is of this reader's format."""
    manifest = json.loads((week_directory / schema.MANIFEST_FILE).read_text())
    if manifest.get("format_version") != schema.FORMAT_VERSION:
        raise ReleaseError(
            f"{week_directory.name} is of release format {manifest.get('format_version')}, "
            f"and this reader reads format {schema.FORMAT_VERSION}"
        )
    return manifest


def checked_day_files(week_directory: pathlib.Path) -> list[pathlib.Path]:
    """The week's trace files, one per day, in day order, each matching the sha256 its manifest records."""
    manifest = checked_manifest(week_directory)
    for file_name, facts in manifest["files"].items():
        if file_sha256(week_directory / file_name) != facts["sha256"]:
            raise ReleaseError(f"{week_directory.name}/{file_name} differs from the sha256 its manifest records")
    return [week_directory / file_name for file_name in sorted(manifest["files"])]


def check_week(week_directory: pathlib.Path) -> None:
    """Stop unless the week is of this reader's format and its tables were converted from its current manifest and
    match the sha256 `tables.json` records."""
    checked_manifest(week_directory)
    tables_file = week_directory / schema.TABLES_FILE
    if not tables_file.exists():
        raise ReleaseError(f"{week_directory.name} is not converted yet; run "
                           f"`python -m release convert --week {week_sunday(week_directory)}`")
    record = json.loads(tables_file.read_text())
    if record["manifest_sha256"] != file_sha256(week_directory / schema.MANIFEST_FILE):
        raise ReleaseError(f"{week_directory.name}'s tables were converted from another manifest; convert it again")
    for file_name, facts in record["files"].items():
        if file_sha256(week_directory / file_name) != facts["sha256"]:
            raise ReleaseError(f"{week_directory.name}/{file_name} differs from the sha256 its tables.json records")


def read_file(week_directory: pathlib.Path, file_name: str) -> pandas.DataFrame:
    """One release file without its block ids, its role tokens flattened; the format's columns must be there."""
    table_schema = schema.SCHEMAS[file_name]
    present = parquet.read_schema(week_directory / file_name).names
    missing = [name for name in table_schema.names if name not in present]
    if missing:
        raise ReleaseError(f"{week_directory.name}/{file_name} lacks the columns {missing}")
    columns = [name for name in table_schema.names if name != BLOCK_IDS]
    return parquet.read_table(week_directory / file_name, columns=columns).flatten().to_pandas()


def file_sha256(path: pathlib.Path) -> str:
    """The sha256 of a file's bytes, as hexadecimal."""
    checksum = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            checksum.update(chunk)
    return checksum.hexdigest()
