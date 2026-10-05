"""A report comparing the export's sessions with a reference export of the same weeks.

The reference is another export's output (`links.parquet`,
`sessions.parquet` and `labels/subagent_links.parquet`). Differences are
expected where a rule changed on purpose; the report counts them so that
each can be looked at. It changes nothing.

Every reference agent session whose root completes in the export gets one
outcome, decided in the week its root completes:
- `same_requests`: a session here with the same root and the same requests;
- `different_requests`: a session here with the same root, other requests;
- `not_an_agent_session_here`: the same root, but not an agent session here;
- `root_is_not_a_root_here`: its root is linked to a parent here;
- `crosses_a_week_end`: its last request completes in a later week, so here
  it is split at the week boundary.
Outcomes are counted per week and per calendar month of the root's completion.
"""

import collections
import datetime
import json
import pathlib

from pyarrow import parquet

from export import paths, settings
from export.log_records import fields


MICROSECONDS_PER_MILLISECOND = 1_000  # the reference's times are in microseconds

# The reference export's files, in its own layout.
REFERENCE_LINKS_FILE = "links.parquet"  # one row per request: its parent and its session
REFERENCE_SESSIONS_FILE = "sessions.parquet"  # one row per session: its root and whether it is an agent session
REFERENCE_SPAWNS_FILE = pathlib.Path("labels") / "subagent_links.parquet"  # one row per spawn call: the child it matched

OUTCOMES = (
    "same_requests",
    "different_requests",
    "not_an_agent_session_here",
    "root_is_not_a_root_here",
    "crosses_a_week_end",
)


def month_of(end_ms: int) -> str:
    """The UTC calendar month of an instant, as `YYYY-MM`."""
    day = fields.UNIX_EPOCH + datetime.timedelta(days=end_ms // fields.MILLISECONDS_PER_DAY)
    return day.isoformat()[:7]


def load_reference(reference_directory: pathlib.Path) -> dict:
    """The reference's agent sessions, keyed by root request id, and its matched spawn pairs."""
    links = parquet.read_table(reference_directory / REFERENCE_LINKS_FILE, columns=["request_id", "end_us", "session"]).to_pylist()
    end_ms_by_request = {}
    requests_by_session = collections.defaultdict(set)
    for row in links:
        end_ms_by_request[row["request_id"]] = row["end_us"] // MICROSECONDS_PER_MILLISECOND
        requests_by_session[row["session"]].add(row["request_id"])
    session_columns = ["session", "root_request_id", "framework", "is_agent_session", "last_end_us"]
    agent_sessions = {}
    for row in parquet.read_table(reference_directory / REFERENCE_SESSIONS_FILE, columns=session_columns).to_pylist():
        root_end_ms = end_ms_by_request.get(row["root_request_id"])
        if row["is_agent_session"] and root_end_ms is not None:
            agent_sessions[row["root_request_id"]] = {
                "harness": row["framework"],
                "requests": requests_by_session[row["session"]],
                "root_week": fields.week_of(root_end_ms),
                "root_month": month_of(root_end_ms),
                "last_week": fields.week_of(row["last_end_us"] // MICROSECONDS_PER_MILLISECOND),
            }
    spawn_columns = ["request_id", "matched", "child_root_request_id"]
    spawn_pairs = set()
    for row in parquet.read_table(reference_directory / REFERENCE_SPAWNS_FILE, columns=spawn_columns).to_pylist():
        if row["matched"]:
            spawn_pairs.add((row["request_id"], row["child_root_request_id"]))
    return {"agent_sessions": agent_sessions, "spawn_pairs": spawn_pairs}


def our_week(week_directory: pathlib.Path) -> dict:
    """This week's sessions: requests per session, agent sessions with harness and month, and spawn pairs."""
    requests_by_session = collections.defaultdict(set)
    agent_sessions = set()
    for row in parquet.read_table(week_directory / paths.CHAINS_FILE).to_pylist():
        requests_by_session[row["session_id"]].add(row["request_id"])
        if row["is_agent_session"]:
            agent_sessions.add(row["session_id"])
    signal_by_request = {}
    for row in parquet.read_table(week_directory / paths.SIGNALS_FILE, columns=["request_id", "harness", "end_ms"]).to_pylist():
        signal_by_request[row["request_id"]] = row
    spawn_pairs = set()
    for row in parquet.read_table(week_directory / paths.SPAWN_MATCHES_FILE).to_pylist():
        if row["child_session_id"] is not None:
            spawn_pairs.add((row["request_id"], row["child_session_id"]))
    return {
        "requests_by_session": requests_by_session,
        "agent_sessions": agent_sessions,
        "harness_by_session": {session: signal_by_request[session]["harness"] for session in agent_sessions},
        "month_by_session": {session: month_of(signal_by_request[session]["end_ms"]) for session in agent_sessions},
        "spawn_pairs": spawn_pairs,
    }


def outcome_of(root: str, session: dict, ours: dict) -> str:
    """What became of one reference agent session here; see the module docstring."""
    if session["last_week"] != session["root_week"]:
        return "crosses_a_week_end"
    if root not in ours["requests_by_session"]:
        return "root_is_not_a_root_here"
    if root not in ours["agent_sessions"]:
        return "not_an_agent_session_here"
    if ours["requests_by_session"][root] == session["requests"]:
        return "same_requests"
    return "different_requests"


def compare_week(sunday: datetime.date, ours: dict, reference: dict) -> tuple[dict, list[tuple[str, str]]]:
    """The week's comparison, and the (month, outcome) of each reference agent session rooted in it."""
    rooted_here = {root: session for root, session in reference["agent_sessions"].items() if session["root_week"] == sunday}
    outcomes = [(session["root_month"], outcome_of(root, session, ours)) for root, session in rooted_here.items()]
    single_week = {root: session for root, session in rooted_here.items() if session["last_week"] == sunday}
    reference_requests = set()
    for session in single_week.values():
        reference_requests |= session["requests"]
    # Spawn matches are compared on the reference's scope: calls made by its single-week agent sessions.
    reference_pairs = {pair for pair in reference["spawn_pairs"] if pair[0] in reference_requests}
    our_pairs = {pair for pair in ours["spawn_pairs"] if pair[0] in reference_requests}
    reference_harnesses = collections.Counter(session["harness"] for session in single_week.values())
    our_harnesses = collections.Counter(ours["harness_by_session"].values())
    week = {
        "week": paths.week_name(sunday),
        "outcomes": dict(collections.Counter(outcome for _, outcome in outcomes)),
        "our_agent_sessions": len(ours["agent_sessions"]),
        "our_agent_sessions_not_in_reference": len(ours["agent_sessions"] - set(rooted_here)),
        "agent_sessions_by_harness": {
            harness: {"reference": reference_harnesses[harness], "ours": our_harnesses[harness]}
            for harness in sorted(set(reference_harnesses) | set(our_harnesses))
        },
        "spawn_matches_on_reference_sessions": {
            "reference": len(reference_pairs),
            "ours": len(our_pairs),
            "in_both": len(reference_pairs & our_pairs),
        },
    }
    return week, outcomes


def month_summary(month: str, outcomes: list[str], ours: int, ours_not_in_reference: int) -> dict:
    """One month's outcome counts and shares, and how many of our agent sessions are new."""
    counts = collections.Counter(outcomes)
    total = len(outcomes)
    return {
        "month": month,
        "reference_agent_sessions": total,
        "outcomes": {outcome: counts[outcome] for outcome in OUTCOMES},
        "percent": {outcome: round(100 * counts[outcome] / total, 1) if total else None for outcome in OUTCOMES},
        "our_agent_sessions": ours,
        "our_agent_sessions_not_in_reference": ours_not_in_reference,
    }


def write_comparison(export_settings: settings.ExportSettings, data: pathlib.Path, reference_directory: pathlib.Path) -> dict:
    """Write `reference_comparison.json` in the export's intermediate folder, for every week and month, and return it."""
    reference = load_reference(reference_directory)
    weeks = []
    outcomes_by_month = collections.defaultdict(list)
    ours_by_month = collections.Counter()
    new_by_month = collections.Counter()
    for sunday in export_settings.weeks:
        ours = our_week(paths.intermediate_week_directory(data, sunday))
        week, outcomes = compare_week(sunday, ours, reference)
        weeks.append(week)
        for month, outcome in outcomes:
            outcomes_by_month[month].append(outcome)
        for session, month in ours["month_by_session"].items():
            ours_by_month[month] += 1
            if session not in reference["agent_sessions"]:
                new_by_month[month] += 1
    months = [
        month_summary(month, outcomes_by_month[month], ours_by_month[month], new_by_month[month])
        for month in sorted(set(outcomes_by_month) | set(ours_by_month))
    ]
    all_outcomes = []
    for outcomes in outcomes_by_month.values():
        all_outcomes.extend(outcomes)
    total = month_summary("all", all_outcomes, sum(ours_by_month.values()), sum(new_by_month.values()))
    report = {"reference": str(reference_directory), "total": total, "months": months, "weeks": weeks}
    with open(paths.intermediate_directory(data) / paths.REFERENCE_COMPARISON_FILE, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report
