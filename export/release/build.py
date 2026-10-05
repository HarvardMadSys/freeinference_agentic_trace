"""The release step: one week's private files to its anonymous release.

Reads the week's folder (`data/export/intermediate/<week>/`): the requests
and sessions, the digests, the transitions, the responses in
`extract.parquet` and the anonymised tool calls. Builds the week's three
tables in memory, checks them, and writes them as
`data/v1/release/<week>/traces/<day>.jsonl`, one JSON line per trace
(`release.trace_lines`), and `manifest.json`. The tables the analysis reads are
converted back from the lines. No text ships.
"""

import collections
import dataclasses
import pathlib
import shutil

import pyarrow
from pyarrow import compute, parquet

from release import schema, trace_lines
from export import parallel, paths, configs
from export.privacy import privacy_check, pseudonyms
from export.release import exclusions, requests_table, tool_calls_table


def release_session_rows(week_sessions: list[dict], step_of_request: dict[str, int], salt: bytes) -> list[dict]:
    """The release's session rows, ordered by their hashed id; `step_of_request` gives a week request's step."""
    rows = []
    for session in week_sessions:
        spawner = session["spawned_by_session_id"]
        rows.append(
            {
                "session_id": pseudonyms.session_hash(session["session_id"], salt),
                "user_id": pseudonyms.user_hash(session["user_id"], salt),
                "harness": session["harness"],
                "parent_session_id": pseudonyms.session_hash(spawner, salt) if spawner is not None else None,
                "spawned_by_step": step_of_request[session["spawned_by_request_id"]] if spawner is not None else None,
                "spawned_by_call_index": session["spawned_by_call_index"] if spawner is not None else None,
            }
        )
    return sorted(rows, key=lambda row: row["session_id"])


def raw_ids(week_requests: list[dict], week_sessions: list[dict]) -> set[str]:
    """Every raw request, session and account id of the week, for the privacy check."""
    found = set()
    for request in week_requests:
        found.update({request["request_id"], request["session_id"], request["user_id"]})
    for session in week_sessions:
        found.update({session["session_id"], session["trace_id"], session["user_id"]})
    found.discard(None)
    return found


def reasons_excluded(week_sessions: list[dict], excluded: exclusions.Exclusions, salt: bytes) -> dict[str, str | None]:
    """Why each of the week's sessions is left out, or None: its own reason, else the nearest excluded spawner's, so a
    subagent of an excluded session is excluded with it."""
    own = {session["session_id"]: exclusions.reason_excluded(session, excluded, salt) for session in week_sessions}
    spawner_of = {session["session_id"]: session["spawned_by_session_id"] for session in week_sessions}
    reasons = {}
    for session_id, reason in own.items():
        spawner = spawner_of[session_id]
        while reason is None and spawner is not None:
            reason = own.get(spawner)
            spawner = spawner_of.get(spawner)
        reasons[session_id] = reason
    return reasons


@dataclasses.dataclass(frozen=True)
class Kept:
    """What the exclusions leave of the week: its requests and sessions, and the excluded sessions counted by reason."""

    requests: list[dict]
    sessions: list[dict]
    excluded_by_reason: dict[str, int]


def kept_by_exclusions(week_requests: list[dict], week_sessions: list[dict], excluded: exclusions.Exclusions,
                       salt: bytes) -> Kept:
    """The requests and sessions the exclusions keep, and the excluded sessions counted by reason."""
    reasons = reasons_excluded(week_sessions, excluded, salt)
    kept_sessions = [session for session in week_sessions if reasons[session["session_id"]] is None]
    kept_requests = [request for request in week_requests if reasons[request["session_id"]] is None]
    counts = collections.Counter(reason for reason in reasons.values() if reason is not None)
    return Kept(requests=kept_requests, sessions=kept_sessions, excluded_by_reason=dict(sorted(counts.items())))


def week_manifest(week_name: str, salt: bytes, excluded_sessions: dict[str, int], distinct_block_ids: int,
                  files: dict[str, dict]) -> dict:
    """The week's manifest: its format, tokenizer, salt series, excluded sessions, distinct block ids and trace files."""
    return {
        "format_version": schema.FORMAT_VERSION,
        "week": week_name,
        "tokenizer": schema.TOKENIZER,
        "block_tokens": schema.BLOCK_TOKENS,
        "salt_series": pseudonyms.salt_series(salt),
        "excluded_sessions": excluded_sessions,
        "distinct_block_ids": distinct_block_ids,
        "files": files,
    }


def week_tables(week: paths.Week, kept: Kept, step_of_request: dict[str, int], tables: configs.RuleTables,
                keys: pseudonyms.PrivateKeys) -> dict[str, pyarrow.Table]:
    """The week's three release tables, from its private files."""
    request_ids = [request["request_id"] for request in kept.requests]
    facts = requests_table.read_request_facts(week, request_ids, tables.public_words)
    categories_of = requests_table.call_categories(week.folder / paths.TOOL_CALLS_FILE)
    rows = requests_table.request_rows(kept.requests, facts, categories_of, keys)
    request_keys = {row["request_id"]: schema.RequestKey(row["session_id"], row["step"]) for row in rows}
    return {
        schema.SESSIONS_FILE: pyarrow.Table.from_pylist(
            release_session_rows(kept.sessions, step_of_request, keys.salt), schema=schema.SESSIONS_SCHEMA),
        schema.REQUESTS_FILE: requests_table.requests_table(rows, week.folder / paths.DIGESTS_FILE),
        schema.TOOL_CALLS_FILE: tool_calls_table.tool_calls_table(
            week.folder / paths.TOOL_CALLS_FILE, request_keys, requests_table.tool_latencies(rows),
            tables.public_words, tables.tool_tables),
    }


def write_release_week(week: paths.Week, tables: configs.RuleTables, keys: pseudonyms.PrivateKeys) -> dict:
    """Build, check and write one release week, and return its manifest."""
    all_requests = parquet.read_table(week.folder / paths.REQUESTS_FILE,
                                      columns=requests_table.WEEK_REQUEST_COLUMNS).to_pylist()
    all_sessions = parquet.read_table(week.folder / paths.SESSIONS_FILE).to_pylist()
    kept = kept_by_exclusions(all_requests, all_sessions, tables.exclusions, keys.salt)
    step_of_request = {request["request_id"]: request["step"] for request in all_requests}
    release_tables = week_tables(week, kept, step_of_request, tables, keys)
    privacy_check.check_week(release_tables, raw_ids(all_requests, all_sessions), tables.harness_names, tables.public_words)
    clear_week(week.release_folder)
    released = trace_lines.WeekTables(sessions=release_tables[schema.SESSIONS_FILE],
                                      requests=release_tables[schema.REQUESTS_FILE],
                                      tool_calls=release_tables[schema.TOOL_CALLS_FILE])
    files = trace_lines.write_lines(released, week.release_folder / schema.TRACES_FOLDER, week.name)
    block_ids = release_tables[schema.REQUESTS_FILE]["block_ids"].combine_chunks().values
    distinct = int(compute.max(block_ids).as_py() or 0)  # ids run from 1 up, one per distinct block
    manifest = week_manifest(week.name, keys.salt, kept.excluded_by_reason, distinct, files)
    parallel.write_json(manifest, week.release_folder / schema.MANIFEST_FILE)
    return manifest


def clear_week(release_week: pathlib.Path) -> None:
    """Empty the week's release folder of an earlier release: its trace files, manifest and converted tables."""
    release_week.mkdir(parents=True, exist_ok=True)
    for path in release_week.iterdir():  # the folder itself may be a link into the data home, so only its contents go
        shutil.rmtree(path) if path.is_dir() else path.unlink()
