"""The export's run record, `data/export/intermediate/run.json`.

It says, for each week of the export, whether its weekly log is complete,
partial or missing, its last timestamp, where it came from and its sha256;
and, for the export, the excluded accounts, the code version and the
versions of the rule tables.
"""

import datetime
import json
import pathlib
import subprocess
import tomllib

from export import paths, configs, settings

def week_status(week_manifest: dict | None, sunday: datetime.date) -> str:
    """`missing` without records, `partial` when the records stop before the week's Saturday, else `complete`."""
    if week_manifest is None or week_manifest["lines"] == 0 or week_manifest["latest_timestamp"] is None:
        return "missing"
    saturday = sunday + datetime.timedelta(days=paths.DAYS_AFTER_SUNDAY_TO_SATURDAY)
    if datetime.date.fromisoformat(week_manifest["latest_timestamp"][:10]) < saturday:
        return "partial"
    return "complete"


def week_entry(raw_directory: pathlib.Path, sunday: datetime.date) -> dict:
    """What the run record says about one week."""
    manifest_path = paths.raw_week_manifest(raw_directory, sunday)
    week_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    entry = {"week": paths.week_name(sunday), "status": week_status(week_manifest, sunday)}
    if week_manifest is not None:
        entry["last_timestamp"] = week_manifest["latest_timestamp"]
        entry["lines"] = week_manifest["lines"]
        entry["sha256"] = week_manifest["sha256"]
        entry["source"] = week_manifest["source"]
    return entry


def code_version() -> str:
    """The repository's commit, with `-dirty` when it has uncommitted changes."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=paths.REPOSITORY_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    changes = subprocess.run(
        ["git", "status", "--porcelain"], cwd=paths.REPOSITORY_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    return commit + "-dirty" if changes else commit


def table_version(path: pathlib.Path) -> str:
    """The `version` a rule table declares."""
    with open(path, "rb") as handle:
        return tomllib.load(handle)["version"]


def write_run_record(export_settings: settings.ExportSettings, data: pathlib.Path) -> dict:
    """Write `run.json` in the export's intermediate folder, and return it."""
    record = {
        "weeks": [week_entry(paths.logs_directory(data), sunday) for sunday in export_settings.weeks],
        "excluded_users": sorted(export_settings.excluded_users),
        "code_version": code_version(),
        "harness_table_version": table_version(configs.HARNESS_TABLE),
        "tool_tables_version": table_version(configs.TOOL_CATEGORIES),
    }
    with open(paths.intermediate_directory(data) / paths.RUN_RECORD_FILE, "w") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    return record
