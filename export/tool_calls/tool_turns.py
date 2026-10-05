"""The tool-turn check: every tool turn of a release session follows a step that made a call.

A tool turn brings tool results back to the model, so the step before it
called a tool. Two cases are exempt, each counted:
- `background_job`: the turn carries Pi's notice that a background job, started
  by a call some steps earlier, has finished;
- `unanswered_result`: every fresh result the turn carries has an id that no
  call of its session emitted.
Any other tool turn after a step with no call means a call format the export
does not read; it stops the step, naming the first requests. The counts per
harness are written to `tool_turns.json` beside `tool_calls.parquet`.
"""

import collections
import dataclasses

from export import parallel, paths
from export.log_records import call_ids
from export.sessions import release_sessions


NAMED_REQUESTS = 5  # how many request ids the error names


class ToolTurnWithoutCall(Exception):
    """A tool turn follows a step that made no call, and no exemption holds."""


def session_call_keys(calls: list[dict]) -> dict[str, set[str]]:
    """The keys of every call id each session's calls emitted."""
    keys = collections.defaultdict(set)
    for call in calls:
        keys[call["session_id"]] |= call_ids.emitted_keys(call["call_id"])
    return keys


def answers_no_call(carried_results, call_keys: set[str]) -> bool:
    """Whether the turn carries results with ids, and none of them answers a call of its session."""
    if not carried_results.with_ids:
        return False
    return all(call_ids.replayed_keys(facts.call_id).isdisjoint(call_keys) for facts in carried_results.with_ids)


@dataclasses.dataclass(frozen=True)
class ToolTurnCounts:
    """Per harness, the tool turns checked and each exemption's count; and the tool turns that fail."""

    per_harness: dict[str, dict[str, int]]
    failing: list[release_sessions.ReleaseRequest]


def tool_turn_counts(requests: list[release_sessions.ReleaseRequest], calls: list[dict], carried: dict) -> ToolTurnCounts:
    """Check every tool turn of the release sessions: the step before it made a call, or an exemption holds."""
    called = {call["request_id"] for call in calls}
    call_keys = session_call_keys(calls)
    by_position = {(request.session_id, request.step): request.request_id for request in requests}
    counts = collections.defaultdict(lambda: {"tool_turns": 0, "background_job": 0, "unanswered_result": 0})
    failing = []
    for request in requests:
        if request.turn != "tool":
            continue
        harness_counts = counts[request.harness]
        harness_counts["tool_turns"] += 1
        if by_position[(request.session_id, request.step - 1)] in called:
            continue
        if carried[request.request_id].background_job:
            harness_counts["background_job"] += 1
        elif answers_no_call(carried[request.request_id], call_keys[request.session_id]):
            harness_counts["unanswered_result"] += 1
        else:
            failing.append(request)
    return ToolTurnCounts({harness: dict(value) for harness, value in sorted(counts.items())}, failing)


def check_week_tool_turns(week: paths.Week, requests: list[release_sessions.ReleaseRequest], calls: list[dict],
                          carried: dict) -> dict:
    """Write the week's `tool_turns.json`, and stop when a tool turn follows no call and is not exempt."""
    counts = tool_turn_counts(requests, calls, carried)
    if counts.failing:
        per_harness = collections.Counter(request.harness for request in counts.failing)
        named = [request.request_id for request in counts.failing[:NAMED_REQUESTS]]
        raise ToolTurnWithoutCall(f"{week.name}: {len(counts.failing)} tool turns follow a step with no call "
                                  f"({dict(per_harness)}), such as {named}")
    parallel.write_json(counts.per_harness, week.folder / paths.TOOL_TURNS_FILE)
    return counts.per_harness
