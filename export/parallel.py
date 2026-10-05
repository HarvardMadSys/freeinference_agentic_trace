"""Reading a file in parts, one process each, and writing a file only once it is complete.

A plain JSONL file is split into byte ranges: a line belongs to the range that
holds its first byte, so every line is read by exactly one range, wherever the
split points fall. A Parquet file is split into ranges of row groups. Each
process writes its rows to a part file; the parts are joined in order. A file
is written under a `.partial` name and takes its own name only once it is
complete, so that a reader never sees a half-written file.
"""

import concurrent.futures
import contextlib
import dataclasses
import json
import math
import multiprocessing
import pathlib
from typing import BinaryIO, Iterator

import pyarrow
from pyarrow import parquet

# One process per 4 GB of a weekly log. At about 140 MB/s of JSON a process reads
# its share in under a minute, and a small week does not start dozens of processes.
BYTES_PER_PROCESS = 4_000_000_000


def processes_for_size(size: int, most: int) -> int:
    """One process per BYTES_PER_PROCESS of `size` bytes, rounded up; at least 1 and at most `most`."""
    return max(1, min(most, math.ceil(size / BYTES_PER_PROCESS)))


def split_into_ranges(path: pathlib.Path, count: int) -> list[tuple[int, int]]:
    """`count` consecutive `(start, end)` byte ranges covering the file, of near-equal size."""
    size = path.stat().st_size
    boundaries = [size * index // count for index in range(count + 1)]
    return [(boundaries[index], boundaries[index + 1]) for index in range(count)]


def lines_in_range(handle: BinaryIO, start: int, end: int) -> Iterator[bytes]:
    """The lines whose first byte lies in `[start, end)`, in file order."""
    if start == 0:
        handle.seek(0)
    else:
        # Finish the line holding byte `start - 1`; the next line starts at or after `start`.
        handle.seek(start - 1)
        handle.readline()
    while handle.tell() < end:
        line = handle.readline()
        if not line:
            return
        yield line


def process_pool(processes: int) -> concurrent.futures.ProcessPoolExecutor:
    """A pool of fresh worker processes for reading ranges, started without copying this process's state."""
    return concurrent.futures.ProcessPoolExecutor(
        max_workers=processes, mp_context=multiprocessing.get_context("spawn")
    )


def row_group_ranges(parquet_file: pathlib.Path, count: int) -> list[tuple[int, int]]:
    """`count` consecutive `(first, stop)` ranges of row groups covering the Parquet file, of near-equal size."""
    row_groups = parquet.ParquetFile(parquet_file).num_row_groups
    boundaries = [row_groups * index // count for index in range(count + 1)]
    return [(boundaries[index], boundaries[index + 1]) for index in range(count)]


def parts_directory(temporary_directory: pathlib.Path, step: str, week_name: str) -> pathlib.Path:
    """The folder of one step's part files for one week, created; the step removes it once the parts are joined."""
    directory = temporary_directory / f"{step}_parts" / week_name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def part_files(directory: pathlib.Path, count: int, suffix: str = ".parquet") -> list[pathlib.Path]:
    """One part file per process, numbered in the order the parts are joined."""
    return [directory / f"{index:05d}{suffix}" for index in range(count)]


@dataclasses.dataclass(frozen=True)
class Shard:
    """The part of a file one process reads, and the part file it writes."""

    file: pathlib.Path
    first: int  # the first row group of a Parquet file, or the first byte of a JSONL file
    stop: int  # one past the last
    part_file: pathlib.Path


def row_group_shards(parquet_file: pathlib.Path, directory: pathlib.Path, count: int) -> list[Shard]:
    """`count` shards of a Parquet file, by row groups, each with its part file under `directory`."""
    return [Shard(parquet_file, first, stop, part_file)
            for (first, stop), part_file in zip(row_group_ranges(parquet_file, count), part_files(directory, count))]


def byte_range_shards(jsonl_file: pathlib.Path, directory: pathlib.Path, count: int, suffix: str = ".parquet") -> list[Shard]:
    """`count` shards of a JSONL file, by byte ranges, each with its part file under `directory`."""
    return [Shard(jsonl_file, start, end, part_file)
            for (start, end), part_file in zip(split_into_ranges(jsonl_file, count), part_files(directory, count, suffix))]


PARTIAL_SUFFIX = ".partial"  # a file being written; it takes its own name once it is complete


@contextlib.contextmanager
def completed_file(path: pathlib.Path) -> Iterator[pathlib.Path]:
    """The name to write `path` under; the file is renamed to `path` once the block completes."""
    partial = path.with_name(path.name + PARTIAL_SUFFIX)
    yield partial
    partial.rename(path)


def write_table(table: pyarrow.Table, output: pathlib.Path) -> None:
    """Write a table as one zstd Parquet file, complete before it takes its name."""
    with completed_file(output) as partial:
        parquet.write_table(table, partial, compression="zstd")


def write_rows(rows: list[dict], table_schema: pyarrow.Schema, output: pathlib.Path) -> None:
    """Write rows of `table_schema` as one zstd Parquet file, complete before it takes its name."""
    write_table(pyarrow.Table.from_pylist(rows, schema=table_schema), output)


def write_json(value: object, output: pathlib.Path) -> None:
    """Write a value as indented JSON with a final newline, complete before it takes its name."""
    with completed_file(output) as partial:
        partial.write_text(json.dumps(value, indent=2) + "\n")


def join_parts(parts: list[pathlib.Path], output: pathlib.Path, table_schema: pyarrow.Schema) -> None:
    """Join the part files, in order, into one zstd Parquet file, complete before it takes its name."""
    with completed_file(output) as partial:
        with parquet.ParquetWriter(partial, table_schema, compression="zstd") as writer:
            for part in parts:
                part_file = parquet.ParquetFile(part)
                for row_group in range(part_file.num_row_groups):
                    writer.write_table(part_file.read_row_group(row_group))
