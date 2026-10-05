"""Where the export's files live, and the one record of a week every command works on.

All data is under `data/` in the repository, a symlink to the data home. It
holds two folders: `data/export/`, the export's inputs and working files,
never published, and `data/v1/`, the dataset. Every stage keeps one folder per
week, named `<sunday>_<saturday>`; a `Week` names that folder, the week's
weekly log, its release folder, and how many processes read it.
"""

import dataclasses
import datetime
import pathlib

from export import parallel

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = pathlib.Path(__file__).resolve().parent
SETTINGS_EXAMPLE = PACKAGE / "settings.example.toml"  # the template of private/export.toml

# The maintainer's files, never committed.
EXPORT_SETTINGS = REPOSITORY_ROOT / "private" / "export.toml"  # this export's weeks, process limit and excluded accounts
PROVIDERS = REPOSITORY_ROOT / "private" / "providers.toml"  # each provider's letter
PROVIDERS_EXAMPLE = PACKAGE / "providers.example.toml"  # the template of private/providers.toml
SALT = REPOSITORY_ROOT / "private" / "salt"  # the release salt, kept in a backup

DAYS_AFTER_SUNDAY_TO_SATURDAY = 6  # a week runs Sunday to Saturday

# The files of a week's folder under data/export/intermediate/, in the order the steps write them.
EXTRACT_FILE = "extract.parquet"  # one row per kept request: its fields, its harness and its verbatim bodies
EXTRACT_COUNTS_FILE = "extract_counts.json"  # the week's lines, unparsable lines, drops by reason and requests kept
SIGNALS_FILE = "signals.parquet"  # one row per kept request: the signals that link it to the request it continues
LINKS_FILE = "links.parquet"  # one row per kept request: its parent request and how it was linked, or none
CHAINS_FILE = "chains.parquet"  # one row per kept request: its session, step, turn, and whether it is on the main chain
SPAWN_MATCHES_FILE = "spawn_matches.parquet"  # one row per spawn call on a main chain: the child session it matched
REQUESTS_FILE = "requests.parquet"  # one row per main-chain request of a release session: its small fields, no bodies
SESSIONS_FILE = "sessions.parquet"  # one row per release session: its harness, counts, start, end and spawner
SPAWNS_FILE = "spawns.parquet"  # one row per spawn call of a release session: the child session it matched
COUNTS_FILE = "counts.json"  # the week's requests, sessions and spawn calls, by stage
SYSTEM_PROMPTS_FILE = "system_prompts.parquet"  # one row per kept request with a top-level system prompt
DIGESTS_FILE = "digests.parquet"  # one row per release request: its message facts and its input's block hashes
TRANSITIONS_FILE = "transitions.parquet"  # one row per release request: its transition from the step before, and the cause
TOOL_CALLS_FILE = "tool_calls.parquet"  # one row per tool call of a release request: as sent, decided, and anonymised
TOOL_TURNS_FILE = "tool_turns.json"  # per harness, the tool turns exempt from following a call, by exemption
WEEK_FILES = (
    EXTRACT_FILE, EXTRACT_COUNTS_FILE, SIGNALS_FILE, LINKS_FILE, CHAINS_FILE, SPAWN_MATCHES_FILE, REQUESTS_FILE,
    SESSIONS_FILE, SPAWNS_FILE, COUNTS_FILE, SYSTEM_PROMPTS_FILE, DIGESTS_FILE, TRANSITIONS_FILE, TOOL_CALLS_FILE,
    TOOL_TURNS_FILE,
)

# The export's files over every week, under data/export/intermediate/.
RUN_RECORD_FILE = "run.json"  # each week's status, source and sha256, the excluded accounts, the code and table versions
ANONYMISATION_REPORT_FILE = "tool_anonymisation.json"  # every tool number before and after anonymisation
REFERENCE_COMPARISON_FILE = "reference_comparison.json"  # the export's sessions against a reference export's


class MissingDataDirectory(Exception):
    """The repository has no `data/` link to the data home."""


def week_name(sunday: datetime.date) -> str:
    """The week's folder name: its Sunday and its Saturday, as `YYYY-MM-DD_YYYY-MM-DD`."""
    saturday = sunday + datetime.timedelta(days=DAYS_AFTER_SUNDAY_TO_SATURDAY)
    return f"{sunday.isoformat()}_{saturday.isoformat()}"


def raw_week_file(raw_directory: pathlib.Path, sunday: datetime.date) -> pathlib.Path:
    """A week's log, `api_logs_<sunday>_<saturday>.jsonl`."""
    return raw_directory / f"api_logs_{week_name(sunday)}.jsonl"


def raw_week_manifest(raw_directory: pathlib.Path, sunday: datetime.date) -> pathlib.Path:
    """The manifest beside a week's weekly log."""
    return raw_directory / f"api_logs_{week_name(sunday)}.manifest.json"


def logs_directory(data: pathlib.Path) -> pathlib.Path:
    """The weekly logs, the export's input."""
    return data / "export" / "gateway_logs"


def intermediate_directory(data: pathlib.Path) -> pathlib.Path:
    """The export's private files, one folder per week."""
    return data / "export" / "intermediate"


def intermediate_week_directory(data: pathlib.Path, sunday: datetime.date) -> pathlib.Path:
    """One week's private folder: every file the export's steps write for it."""
    return intermediate_directory(data) / week_name(sunday)


def temporary_directory(data: pathlib.Path) -> pathlib.Path:
    """Where every step writes its temporary files, on the data home's disk."""
    return data / "export" / "tmp"


def release_directory(data: pathlib.Path) -> pathlib.Path:
    """The anonymous release, one folder per week."""
    return data / "v1" / "release"


def release_week_directory(data: pathlib.Path, sunday: datetime.date) -> pathlib.Path:
    """The folder of the anonymous release for one week."""
    return release_directory(data) / week_name(sunday)


def data_directory(repository_root: pathlib.Path = REPOSITORY_ROOT) -> pathlib.Path:
    """The repository's `data/` folder; an error if it does not exist."""
    data = repository_root / "data"
    if not data.is_dir():
        raise MissingDataDirectory(
            f"{data} does not exist: data/export and data/v1/release must link to the data home"
        )
    return data


@dataclasses.dataclass(frozen=True)
class Week:
    """One week of the export: its files, and how many processes a step of it uses."""

    sunday: datetime.date
    name: str  # `<sunday>_<saturday>`, the name of its folders
    log_file: pathlib.Path  # the week's log, the export's input
    log_manifest: pathlib.Path  # the manifest beside it
    folder: pathlib.Path  # the week's private files, under data/export/intermediate/
    release_folder: pathlib.Path  # the week's anonymous release, under data/v1/release/
    temporary_folder: pathlib.Path  # where its steps write their part files
    processes: int  # one per 4 GB of the weekly log, up to the maintainer's most


def week(sunday: datetime.date, data: pathlib.Path, max_processes: int) -> Week:
    """The week of `sunday` in the data home; a week whose weekly log is gone is read by one process."""
    logs = logs_directory(data)
    log_file = raw_week_file(logs, sunday)
    processes = 1
    if log_file.exists():
        processes = parallel.processes_for_size(log_file.stat().st_size, max_processes)
    return Week(
        sunday=sunday,
        name=week_name(sunday),
        log_file=log_file,
        log_manifest=raw_week_manifest(logs, sunday),
        folder=intermediate_week_directory(data, sunday),
        release_folder=release_week_directory(data, sunday),
        temporary_folder=temporary_directory(data),
        processes=processes,
    )
