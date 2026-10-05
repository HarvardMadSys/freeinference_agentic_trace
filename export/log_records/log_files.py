"""The weekly log files: a week's manifest, and cutting weeks out of a longer export.

A manifest records where a week's file came from, its line count, byte count
and sha256, its earliest and latest `timestamp`, and how many lines are not
JSON objects, so that later steps can trust the file.

Cutting takes a longer, plain JSONL export and sends each record to the week
that holds its completion `timestamp`, unchanged and in source order. Only
weeks that have no file yet are written, so a weekly file already in
`data/export/gateway_logs/` is never overwritten. The source is read by
several processes, one byte range each; each writes one part file per week,
and each week's parts are joined in range order into the week's file, whose
manifest is written in the same pass.
"""
import contextlib
import dataclasses
import datetime
import hashlib
import json
import pathlib
import shutil

from export import parallel, paths
from export.log_records import fields


HASH_CHUNK_BYTES = 16 * 1024 * 1024  # read size when hashing a whole file


class WeekFileError(Exception):
    """A week's weekly log holds a record of another week, or disagrees with its manifest."""


@dataclasses.dataclass
class WeekTally:
    """Running counts over the lines of one week's weekly log, in file order."""

    week: datetime.date
    lines: int = 0
    bytes: int = 0
    invalid_json_lines: int = 0
    earliest_ms: int | None = None
    earliest_timestamp: str | None = None
    latest_ms: int | None = None
    latest_timestamp: str | None = None
    checksum: object = dataclasses.field(default_factory=hashlib.sha256)

    def add(self, line: bytes, record: dict | None) -> None:
        """Count one line; `record` is what the line parsed to, or None."""
        self.lines += 1
        self.bytes += len(line)
        self.checksum.update(line)
        if record is None:
            self.invalid_json_lines += 1
            return
        timestamp = record.get("timestamp")
        end_ms = fields.completion_ms(timestamp)
        if end_ms is None:
            return
        if fields.week_of(end_ms) != self.week:
            raise WeekFileError(
                f"record {record.get('request_id')} completed at {timestamp}, "
                f"outside the week {paths.week_name(self.week)}"
            )
        if self.earliest_ms is None or end_ms < self.earliest_ms:
            self.earliest_ms = end_ms
            self.earliest_timestamp = timestamp
        if self.latest_ms is None or end_ms > self.latest_ms:
            self.latest_ms = end_ms
            self.latest_timestamp = timestamp

    def manifest(self, file_name: str, source: dict) -> dict:
        """The manifest of the lines counted so far."""
        return {
            "week": paths.week_name(self.week),
            "file": file_name,
            "source": source,
            "lines": self.lines,
            "bytes": self.bytes,
            "sha256": self.checksum.hexdigest(),
            "earliest_timestamp": self.earliest_timestamp,
            "latest_timestamp": self.latest_timestamp,
            "invalid_json_lines": self.invalid_json_lines,
        }


def build_manifest(week_file: pathlib.Path, week: datetime.date, source: dict) -> dict:
    """Read a whole week's weekly log and return its manifest."""
    tally = WeekTally(week)
    with open(week_file, "rb") as handle:
        for line in handle:
            tally.add(line, fields.parse_line(line))
    return tally.manifest(week_file.name, source)


def write_manifest(manifest: dict, path: pathlib.Path) -> None:
    """Write a manifest as indented JSON."""
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def check_week_file(week_file: pathlib.Path, manifest_path: pathlib.Path) -> dict:
    """The week's manifest; an error unless the file has the size and sha256 it records."""
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    size = week_file.stat().st_size
    if size != manifest["bytes"]:
        raise WeekFileError(
            f"{week_file.name} has {size} bytes, but its manifest records {manifest['bytes']}"
        )
    checksum = file_sha256(week_file)
    if checksum != manifest["sha256"]:
        raise WeekFileError(f"{week_file.name} has a different sha256 than its manifest records")
    return manifest


def file_sha256(path: pathlib.Path) -> str:
    """The sha256 of a file's bytes, as hexadecimal."""
    checksum = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            checksum.update(chunk)
    return checksum.hexdigest()




@dataclasses.dataclass
class SourceTally:
    """What happened to the lines of the source export, or of one byte range of it."""

    lines: int = 0
    unreadable_lines: int = 0  # not a JSON object, or no readable `timestamp`
    lines_of_other_weeks: int = 0  # records of weeks that are not being cut

    def add(self, other: "SourceTally") -> None:
        """Add another range's counts to these."""
        self.lines += other.lines
        self.unreadable_lines += other.unreadable_lines
        self.lines_of_other_weeks += other.lines_of_other_weeks


def weeks_to_cut(weeks: tuple[datetime.date, ...], raw_directory: pathlib.Path) -> list[datetime.date]:
    """The weeks that have no weekly log yet."""
    return [week for week in weeks if not paths.raw_week_file(raw_directory, week).exists()]


def cut_weeks(
    source: pathlib.Path,
    weeks: list[datetime.date],
    raw_directory: pathlib.Path,
    temporary_directory: pathlib.Path,
    processes: int,
) -> SourceTally:
    """Write each of `weeks`, and its manifest, from the records of `source`."""
    parts_directory = temporary_directory / "cut_parts"
    # Each shard's part file is a folder, holding one part per week; the week's parts are joined in shard order.
    shards = parallel.byte_range_shards(source, parts_directory, processes, suffix="")
    with parallel.process_pool(processes) as pool:
        range_jobs = [pool.submit(cut_range, shard, weeks) for shard in shards]
        source_tally = SourceTally()
        for job in range_jobs:
            source_tally.add(job.result())
        source_description = {
            "kind": "cut from a longer export",
            "path": str(source.resolve()),
            "lines": source_tally.lines,
            "unreadable_lines": source_tally.unreadable_lines,
        }
        join_jobs = [pool.submit(join_week, week, shards, raw_directory, source_description) for week in weeks]
        for job in join_jobs:
            job.result()
    shutil.rmtree(parts_directory)
    return source_tally


def cut_range(shard: parallel.Shard, weeks: list[datetime.date]) -> SourceTally:
    """Route the lines of one byte range of the source into the range's part of each week."""
    range_tally = SourceTally()
    with contextlib.ExitStack() as open_files:
        outputs = {}
        for week in weeks:
            part = week_part(shard, week)
            part.parent.mkdir(parents=True, exist_ok=True)
            outputs[week] = open_files.enter_context(open(part, "wb"))
        with open(shard.file, "rb") as handle:
            for line in parallel.lines_in_range(handle, shard.first, shard.stop):
                route_line(ensure_line_end(line), range_tally, outputs)
    return range_tally


def route_line(line: bytes, range_tally: SourceTally, outputs: dict) -> None:
    """Write one source line to its week's part file, or count why it was not written."""
    range_tally.lines += 1
    record = fields.parse_line(line)
    end_ms = None
    if record is not None:
        end_ms = fields.completion_ms(record.get("timestamp"))
    if end_ms is None:
        range_tally.unreadable_lines += 1
        return
    week = fields.week_of(end_ms)
    if week not in outputs:
        range_tally.lines_of_other_weeks += 1
        return
    outputs[week].write(line)


def ensure_line_end(line: bytes) -> bytes:
    """The line with its newline; only a source's last line can lack one."""
    if line.endswith(b"\n"):
        return line
    return line + b"\n"


def join_week(week: datetime.date, shards: list[parallel.Shard], raw_directory: pathlib.Path,
              source_description: dict) -> None:
    """Join a week's parts, in shard order, into its weekly log, and write the log's manifest."""
    week_file = paths.raw_week_file(raw_directory, week)
    week_tally = WeekTally(week)
    with parallel.completed_file(week_file) as partial, open(partial, "wb") as output:
        for shard in shards:
            with open(week_part(shard, week), "rb") as part:
                for line in part:
                    output.write(line)
                    week_tally.add(line, fields.parse_line(line))
    week_manifest = week_tally.manifest(week_file.name, source_description)
    write_manifest(week_manifest, paths.raw_week_manifest(raw_directory, week))


def week_part(shard: parallel.Shard, week: datetime.date) -> pathlib.Path:
    """Where one shard's records of one week are written: a file named after the week in the shard's folder."""
    return shard.part_file / f"{paths.week_name(week)}.jsonl"
