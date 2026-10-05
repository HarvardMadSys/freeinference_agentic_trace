"""The simulation's weekly files: their names, and every quantity summed over the weeks before any figure divides.

The figures read these, never the weekly files, so that every ratio is taken
over all weeks together. Each week's `sources.json` names, per file, the
sha256 of the week's manifest it was computed from; the manifest holds every
trace file's sha256, so it identifies the traces' content. A file is read
only when every release week's was computed from that week's current traces.
Tables converted again from the same traces, by any Parquet writer, leave a
week current.
"""

import json
import pathlib

import pandas

from release import reader
from release import schema

TALLY_COLUMNS = ["hit_tokens", "miss_tokens", "reusable_token_seconds", "dead_token_seconds"]
TOTAL_COLUMNS = ["active_token_seconds", "span_seconds", "output_tokens", "requests"]
TALLIES_FILE = "tallies.parquet"  # per scope, policy and TTL: hit and missed tokens, reusable and dead token-seconds
TOTALS_FILE = "totals.parquet"  # per scope: active token-seconds, the span, output tokens and requests
REQUEST_MISSES_FILE = "request_misses.parquet"  # per request: the tokens it missed with no TTL under ttl_only
CHEAPER_READS_FILE = "cheaper_reads.parquet"  # per TTL and price cut: the cheaper-reads model's sums
CHEAPER_READS_COLUMNS = ["today_prefilled_tokens", "today_cached_tokens", "prefilled_tokens", "cached_tokens",
                         "output_tokens", "reductions", "skipped_reductions", "forced_compactions"]
SOURCES_FILE = "sources.json"  # per file of a simulated week, the sha256 of the manifest it was computed from
SIMULATION_FILES = (TALLIES_FILE, TOTALS_FILE, REQUEST_MISSES_FILE, CHEAPER_READS_FILE)  # what `simulate` writes


class SimulationMissing(Exception):
    """No week has been simulated."""


class SimulationStale(Exception):
    """A release week's file was not computed from the week's current traces."""


def has_weeks(simulation_directory: pathlib.Path) -> bool:
    """Whether any week has been simulated into the folder."""
    return any(simulation_directory.glob("*/tallies.parquet"))


def has_cheaper_reads(simulation_directory: pathlib.Path) -> bool:
    """Whether any week's cheaper-reads model has run into the folder."""
    return any(simulation_directory.glob(f"*/{CHEAPER_READS_FILE}"))


def recorded_sources(simulated_week: pathlib.Path) -> dict:
    """A simulated week's `sources.json`: per file, the manifest's sha256 it was computed from; empty without one."""
    path = simulated_week / SOURCES_FILE
    return json.loads(path.read_text()) if path.exists() else {}


def traces_source(week_directory: pathlib.Path) -> dict[str, str]:
    """What identifies a release week's traces: the sha256 of its manifest, which names every trace file's sha256."""
    return {schema.MANIFEST_FILE: reader.file_sha256(week_directory / schema.MANIFEST_FILE)}


def record_sources(simulated_week: pathlib.Path, week_directory: pathlib.Path, file_names: list[str]) -> None:
    """Record in the simulated week's `sources.json` that the files were computed from the week's current traces."""
    sources = recorded_sources(simulated_week)
    for file_name in file_names:
        sources[file_name] = traces_source(week_directory)
    (simulated_week / SOURCES_FILE).write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n")


def stale_weeks(release_directory: pathlib.Path, simulation_directory: pathlib.Path) -> list[pathlib.Path]:
    """The release weeks whose simulation is missing or was computed from other traces, in week order."""
    stale = []
    for week_directory in reader.find_weeks(pathlib.Path(release_directory).expanduser(), None):
        sources = recorded_sources(simulation_directory / week_directory.name)
        current = traces_source(week_directory)
        if any(sources.get(file_name) != current for file_name in SIMULATION_FILES):
            stale.append(week_directory)
    return stale


def week_files(simulation_directory: pathlib.Path, release_directory: pathlib.Path, file_name: str) -> list[pathlib.Path]:
    """The given file of every release week, in week order; stop unless each was computed from its week's current
    traces."""
    paths = []
    for week_directory in reader.find_weeks(pathlib.Path(release_directory).expanduser(), None):
        simulated_week = simulation_directory / week_directory.name
        if recorded_sources(simulated_week).get(file_name) != traces_source(week_directory):
            raise SimulationStale(f"{week_directory.name}'s {file_name} was not computed from its current traces; run "
                                  f"`python -m simulation --week {reader.week_sunday(week_directory)}`")
        if (simulated_week / file_name).exists():  # a week without Agentic requests records its source, no cheaper-reads file
            paths.append(simulated_week / file_name)
    if not paths:
        raise SimulationMissing(f"{simulation_directory} holds no simulated week; run `python -m simulation`")
    return paths


def tallies(simulation_directory: pathlib.Path, release_directory: pathlib.Path) -> pandas.DataFrame:
    """Per scope, policy and TTL, the tallies summed over the weeks."""
    paths = week_files(simulation_directory, release_directory, TALLIES_FILE)
    weekly = pandas.concat([pandas.read_parquet(path) for path in paths])
    return weekly.groupby(["scope", "policy", "ttl"], sort=False, as_index=False)[TALLY_COLUMNS].sum()


def totals(simulation_directory: pathlib.Path, release_directory: pathlib.Path) -> pandas.DataFrame:
    """Per scope, the totals summed over the weeks, indexed by scope."""
    paths = week_files(simulation_directory, release_directory, TOTALS_FILE)
    weekly = pandas.concat([pandas.read_parquet(path) for path in paths])
    return weekly.groupby("scope")[TOTAL_COLUMNS].sum()


def request_misses(simulation_directory: pathlib.Path, release_directory: pathlib.Path) -> pandas.DataFrame:
    """Per request, the tokens it missed with no TTL under `ttl_only`; every week's rows together."""
    paths = week_files(simulation_directory, release_directory, REQUEST_MISSES_FILE)
    return pandas.concat([pandas.read_parquet(path) for path in paths], ignore_index=True)


def cheaper_reads(simulation_directory: pathlib.Path, release_directory: pathlib.Path) -> pandas.DataFrame:
    """Per TTL and read price cut, the cheaper-reads model's sums and counts summed over the weeks."""
    paths = week_files(simulation_directory, release_directory, CHEAPER_READS_FILE)
    weekly = pandas.concat([pandas.read_parquet(path) for path in paths])
    return weekly.groupby(["ttl", "read_price_cut"], sort=False, as_index=False)[CHEAPER_READS_COLUMNS].sum()
