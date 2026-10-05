"""Extract: one week's weekly log to its kept requests.

Every line of the week is counted exactly once: as unparsable, as an
unreadable record, under the first filter it fails, or as kept. Kept requests
are written with their fields and harness to `extract.parquet`.
"""

import collections
import datetime
import pathlib
import shutil

import pyarrow
from pyarrow import compute, parquet

from export import parallel, paths
from export.log_records import fields, harness, log_files, messages



KEPT_STATUS_CODE = 200  # only successful requests can belong to a session

ROWS_PER_WRITE = 500  # kept requests held in memory before a write; a request averages about 0.5 MB

KEPT = "requests_kept"
UNPARSABLE = "unparsable_line"
DROP_REASONS = ("unreadable_record", "excluded_user", "status_not_200", "harness_not_shortlisted")

EXTRACT_SCHEMA = pyarrow.schema(
    [
        ("request_id", pyarrow.string()),
        ("user_id", pyarrow.string()),
        ("harness", pyarrow.string()),
        ("end_ms", pyarrow.int64()),
        ("start_ms", pyarrow.int64()),
        ("latency_ms", pyarrow.int64()),
        ("ttft_ms", pyarrow.int64()),
        ("stream", pyarrow.bool_()),
        ("model_id", pyarrow.string()),
        ("provider", pyarrow.string()),
        ("prompt_tokens", pyarrow.int64()),
        ("completion_tokens", pyarrow.int64()),
        ("reasoning_tokens", pyarrow.int64()),
        ("cache_read_tokens", pyarrow.int64()),
        ("cache_write_tokens", pyarrow.int64()),
        # Bodies can be megabytes each, more than a plain string column holds per batch.
        ("prompt", pyarrow.large_string()),
        ("response", pyarrow.large_string()),
        ("tools", pyarrow.large_string()),
    ]
)


def classify_line(line: bytes, excluded_users: frozenset[str], harnesses: tuple) -> tuple[str, dict | None]:
    """What happens to one line: its count name, and its kept row when it is kept."""
    record = fields.parse_line(line)
    if record is None:
        return UNPARSABLE, None
    request_id = fields.read_string(record.get("request_id"))
    end_ms = fields.completion_ms(record.get("timestamp"))
    if not request_id or end_ms is None:
        return "unreadable_record", None
    if fields.read_string(record.get("user_id")) in excluded_users:
        return "excluded_user", None
    if fields.read_integer(record.get("status_code")) != KEPT_STATUS_CODE:
        return "status_not_200", None
    input_messages = messages.messages_list(fields.read_body(record.get("prompt")))
    name = harness.harness_name(input_messages, fields.user_agent(record), harnesses)
    if name == harness.UNRECOGNISED_HARNESS:
        return "harness_not_shortlisted", None
    row = fields.kept_fields(record, end_ms)
    row["harness"] = name
    return KEPT, row


def extract_week(week: paths.Week, excluded_users: frozenset[str], harnesses: tuple) -> dict:
    """Write the week's `extract.parquet` and `extract_counts.json`, and return the counts.

    The weekly log is read by the week's processes, one byte range each; their
    parts are joined in range order, so the rows keep the file's order.
    """
    week_manifest = log_files.check_week_file(week.log_file, week.log_manifest)
    parts_directory = parallel.parts_directory(week.temporary_folder, "extract", week.name)
    shards = parallel.byte_range_shards(week.log_file, parts_directory, week.processes)
    counts = collections.Counter()
    with parallel.process_pool(week.processes) as pool:
        jobs = [pool.submit(extract_range, shard, excluded_users, harnesses) for shard in shards]
        for job in jobs:
            counts.update(job.result())
    lines = sum(counts.values())
    if lines != week_manifest["lines"]:
        raise log_files.WeekFileError(f"read {lines} lines, but the manifest records {week_manifest['lines']}")
    parallel.join_parts([shard.part_file for shard in shards], week.folder / paths.EXTRACT_FILE, EXTRACT_SCHEMA)
    shutil.rmtree(parts_directory)
    week_counts = counts_summary(week.sunday, counts)
    parallel.write_json(week_counts, week.folder / paths.EXTRACT_COUNTS_FILE)
    return week_counts


def extract_range(shard: parallel.Shard, excluded_users: frozenset[str], harnesses: tuple) -> collections.Counter:
    """Classify the lines of one byte range of the weekly log, writing its kept rows, uncompressed, to the shard's
    part file."""
    counts = collections.Counter()
    with parquet.ParquetWriter(shard.part_file, EXTRACT_SCHEMA, compression="none") as writer:
        rows = []
        with open(shard.file, "rb") as handle:
            for line in parallel.lines_in_range(handle, shard.first, shard.stop):
                outcome, row = classify_line(line, excluded_users, harnesses)
                counts[outcome] += 1
                if row is not None:
                    rows.append(row)
                if len(rows) == ROWS_PER_WRITE:
                    write_rows(writer, rows)
                    rows = []
        write_rows(writer, rows)
    return counts


def write_rows(writer: parquet.ParquetWriter, rows: list[dict]) -> None:
    """Write a batch of kept rows."""
    if rows:
        writer.write_table(pyarrow.Table.from_pylist(rows, schema=EXTRACT_SCHEMA))


def counts_summary(week: datetime.date, counts: collections.Counter) -> dict:
    """The week's line counts, so that `lines = unparsable_line + dropped + requests_kept`."""
    return {
        "week": paths.week_name(week),
        "lines": sum(counts.values()),
        UNPARSABLE: counts[UNPARSABLE],
        "dropped": {reason: counts[reason] for reason in DROP_REASONS},
        KEPT: counts[KEPT],
    }


def prompts_of(extract_file: pathlib.Path, request_ids: list[str]) -> dict[str, str | None]:
    """The prompts of the given requests, read from the row groups that hold them."""
    return column_of(extract_file, request_ids, "prompt")


def column_of(extract_file: pathlib.Path, request_ids: list[str], column: str) -> dict[str, str | None]:
    """One column of the given requests, by request id, read from the row groups that hold them."""
    wanted = pyarrow.array(sorted(request_ids), type=pyarrow.string())
    extract = parquet.ParquetFile(extract_file)
    values = {}
    for row_group in range(extract.num_row_groups):
        ids = extract.read_row_group(row_group, columns=["request_id"])["request_id"]
        if not compute.any(compute.is_in(ids, value_set=wanted)).as_py():
            continue
        table = extract.read_row_group(row_group, columns=["request_id", column])
        table = table.filter(compute.is_in(table["request_id"], value_set=wanted))
        values.update(zip(table["request_id"].to_pylist(), table[column].to_pylist()))
    return values
