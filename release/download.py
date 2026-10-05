"""Downloading the release from Hugging Face into the layout the analysis reads.

The Hub repository keeps, per week, `release/<week>/manifest.json`, the tables
under `release/<week>/processed/`, and the trace files compressed as
`release/<week>/traces/<day>.jsonl.zst`. A download fetches what is asked for
into a hub folder and lays it out as `data/v1/release`: the tables (`tables`),
the trace files decompressed (`traces`), or both. Each table is then checked
against `tables.json` and each trace file against the manifest, as the reader
does. The simulation ships with this repository, under `data/v1/simulation`.
"""

import os
import pathlib
import shutil
import subprocess

from release import reader
from release import schema

REPOSITORY = "harvardMadsys/freeinference_agentic_trace"  # the Hub repository that holds the release
HUB_RELEASE_FOLDER = "release"  # the Hub's week folders
HUB_TABLES_FOLDER = "processed"  # the Hub's name for a week's tables
COMPRESSED_SUFFIX = ".zst"  # the Hub's trace files are zstd-compressed
PARTIAL_SUFFIX = ".partial"  # a trace file being decompressed; it takes its own name once complete
TABLE_FILES = (schema.SESSIONS_FILE, schema.REQUESTS_FILE, schema.TOOL_CALLS_FILE, schema.TABLES_FILE)


class DownloadError(Exception):
    """The download cannot be laid out: a tool is missing, or a week is incomplete."""


def hub_patterns(tables: bool, traces: bool, weeks: list[str] | None) -> list[str]:
    """The Hub paths to fetch: every chosen week's manifest, and its tables or its trace files."""
    week_globs = [f"{week}_*" for week in weeks] if weeks else ["*"]
    patterns = []
    for week in week_globs:
        patterns.append(f"{HUB_RELEASE_FOLDER}/{week}/{schema.MANIFEST_FILE}")
        if tables:
            patterns.append(f"{HUB_RELEASE_FOLDER}/{week}/{HUB_TABLES_FOLDER}/*")
        if traces:
            patterns.append(f"{HUB_RELEASE_FOLDER}/{week}/{schema.TRACES_FOLDER}/*")
    return patterns


def fetch(repo: str, hub_directory: pathlib.Path, patterns: list[str]) -> pathlib.Path:
    """Fetch the matching files of the Hub repository into `hub_directory`, resuming what is there; return it."""
    try:
        import huggingface_hub
    except ImportError:
        raise DownloadError("downloading needs the huggingface_hub package: pip install -e '.[huggingface]'")
    return pathlib.Path(huggingface_hub.snapshot_download(repo_id=repo, repo_type="dataset", local_dir=hub_directory,
                                                           allow_patterns=patterns))


def place(source: pathlib.Path, destination: pathlib.Path) -> None:
    """Put a fetched file at its place in `data/v1`: a hard link when the disks allow it, else a copy."""
    if not source.exists():
        raise DownloadError(f"{source} was not downloaded; the week is incomplete on the Hub or the fetch stopped")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def decompress(compressed: pathlib.Path, output: pathlib.Path) -> None:
    """Decompress one trace file with `zstd`, writing it complete or not at all."""
    if shutil.which("zstd") is None:
        raise DownloadError("decompressing the trace files needs the zstd command; install zstd, or download --tables only")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + PARTIAL_SUFFIX)
    subprocess.run(["zstd", "-d", "-q", "-f", str(compressed), "-o", str(partial)], check=True)
    os.replace(partial, output)


def lay_out_week(hub_week: pathlib.Path, release_week: pathlib.Path, tables: bool, traces: bool) -> None:
    """One fetched week laid out as `data/v1/release` holds it, then checked as the reader checks it."""
    place(hub_week / schema.MANIFEST_FILE, release_week / schema.MANIFEST_FILE)
    if tables:
        for name in TABLE_FILES:
            place(hub_week / HUB_TABLES_FOLDER / name, release_week / name)
        reader.check_week(release_week)
    if traces:
        for compressed in sorted((hub_week / schema.TRACES_FOLDER).glob(f"*{COMPRESSED_SUFFIX}")):
            decompress(compressed, release_week / schema.TRACES_FOLDER / compressed.name.removesuffix(COMPRESSED_SUFFIX))
        reader.checked_day_files(release_week)


def download(repo: str, hub_directory: pathlib.Path, release_directory: pathlib.Path, tables: bool, traces: bool,
             weeks: list[str] | None) -> list[str]:
    """Fetch the chosen weeks and lay them out; return the week folders laid out, oldest first."""
    hub = fetch(repo, hub_directory, hub_patterns(tables, traces, weeks))
    fetched = sorted(path.name for path in (hub / HUB_RELEASE_FOLDER).iterdir() if (path / schema.MANIFEST_FILE).exists())
    if weeks:
        fetched = [week for week in fetched if reader.week_sunday(pathlib.Path(week)) in weeks]
        missing = sorted(set(weeks) - {reader.week_sunday(pathlib.Path(week)) for week in fetched})
        if missing:
            raise DownloadError(f"no week on the Hub starts on {', '.join(missing)}")
    for week in fetched:
        lay_out_week(hub / HUB_RELEASE_FOLDER / week, release_directory / week, tables, traces)
    return fetched
