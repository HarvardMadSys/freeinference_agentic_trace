"""Relabel: decide again, with the current harness table, the harness of a week's
sessions that carry one harness name, without reading the weekly log again.

It is for a harness table that splits an entry, such as `oh-my-pi` out of
`pi`, or drops a user-agent rule, such as Claude Code's. A session's harness
is its first request's harness, and a session's id is its first request's id.
The extract keeps each request's prompt but not its user agent, so the harness
is decided again from phrases alone. A first request whose phrases no longer
name a harness makes the session `other-agent`, which the release leaves out,
when its harness has no user-agent rule; when it has one, the user agent may
have named it, and the step stops.

Only `sessions.parquet` is rewritten. `build --from release-sessions` takes
each session's harness from the extract again, so run `relabel` after it
whenever the harness table has changed since the extract.
"""

import collections

from pyarrow import parquet

from export import parallel, paths
from export.log_records import harness, messages
from export.sessions import extract, release_sessions


class RelabelError(Exception):
    """A session's new harness cannot be decided, or would change the release sessions."""


def harness_by_phrases(prompt: str | None, harnesses: tuple[harness.Harness, ...]) -> str | None:
    """The harness a request's phrases name, or None."""
    window = harness.intro_window(messages.messages_list(prompt))
    return harness.harness_by_earliest_phrase(window, harnesses)


def has_user_agent_rule(name: str, harnesses: tuple[harness.Harness, ...]) -> bool:
    """Whether the table's entry for this harness recognises it by user agent too."""
    return any(entry.name == name and entry.user_agents for entry in harnesses)


def new_harness(session: dict, prompt: str | None, harnesses: tuple[harness.Harness, ...]) -> str:
    """The session's harness under the current table; stops when it cannot be decided or leaves the release sessions."""
    name = harness_by_phrases(prompt, harnesses)
    if name is None and has_user_agent_rule(session["harness"], harnesses):
        raise RelabelError(
            f"session {session['session_id']}: its first request's phrases name no harness, "
            f"and {session['harness']}'s user agent is not kept to decide it"
        )
    if name is None:
        name = harness.UNRECOGNISED_HARNESS
    if (name == harness.BENCHMARK_HARNESS) != (session["harness"] == harness.BENCHMARK_HARNESS):
        raise RelabelError(
            f"session {session['session_id']}: {session['harness']} to {name} moves it into or out of the "
            f"benchmark, which changes the release sessions; run `build --from release-sessions` instead"
        )
    return name


def relabel_week(week: paths.Week, name: str, harnesses: tuple[harness.Harness, ...]) -> dict[str, int]:
    """Decide again the harness of the week's sessions whose harness is `name`, rewriting `sessions.parquet`; count
    the harnesses decided."""
    sessions = parquet.read_table(week.folder / paths.SESSIONS_FILE).to_pylist()
    chosen = [session for session in sessions if session["harness"] == name]
    prompts = extract.prompts_of(week.folder / paths.EXTRACT_FILE, [session["session_id"] for session in chosen])
    counts = collections.Counter()
    for session in chosen:
        session["harness"] = new_harness(session, prompts.get(session["session_id"]), harnesses)
        counts[session["harness"]] += 1
    parallel.write_rows(sessions, release_sessions.SESSIONS_SCHEMA, week.folder / paths.SESSIONS_FILE)
    return dict(counts)
