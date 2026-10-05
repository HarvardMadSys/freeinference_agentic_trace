"""The steps that build a week, in the order they run, and what each reads and writes.

`build_week` runs the steps from a named one to the last. Before a step runs
it checks that every week file the step reads exists, and otherwise stops with
the command that writes it. The command line and the tests run this table, so
the order of the steps lives here only.
"""

import dataclasses
import typing

from export import paths, configs
from export.log_records import log_files
from export.request_inputs import digests, system_prompts, transitions
from export.sessions import chains, extract, link, release_sessions, signals, spawns
from export.tool_calls import calls


class MissingInputError(Exception):
    """A step's input is missing; the message names the command that writes it."""


@dataclasses.dataclass(frozen=True)
class BuildInputs:
    """What the steps need besides the week: the rule tables, the accounts the extract leaves out, and the salt."""

    tables: configs.RuleTables
    excluded_users: frozenset[str]
    salt: bytes


@dataclasses.dataclass(frozen=True)
class Step:
    """One step of building a week: its name on the command line, what it does, the week files it reads and
    writes, and the function that runs it and returns a sentence saying what it did."""

    name: str
    help: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    run: typing.Callable[[paths.Week, BuildInputs], str]


def run_manifest(week: paths.Week, inputs: BuildInputs) -> str:
    """Write the weekly log's manifest when it is absent; a manifest written by `cut-weeks` is kept."""
    if week.log_manifest.exists():
        return f"{week.name}: the weekly log's manifest is present"
    source = {"kind": "gateway weekly export", "path": str(week.log_file.resolve())}
    week_manifest = log_files.build_manifest(week.log_file, week.sunday, source)
    log_files.write_manifest(week_manifest, week.log_manifest)
    return f"{week.name}: the weekly log has {week_manifest['lines']} lines, {week_manifest['bytes']} bytes"


def run_extract(week: paths.Week, inputs: BuildInputs) -> str:
    counts = extract.extract_week(week, inputs.excluded_users, inputs.tables.harnesses)
    return f"{week.name}: {counts['lines']} lines read, {counts[extract.KEPT]} requests kept"


def run_signals(week: paths.Week, inputs: BuildInputs) -> str:
    requests = signals.write_week_signals(week, inputs.tables.tool_tables.spawn_tools)
    return f"{week.name}: linking signals of {requests} requests"


def run_link(week: paths.Week, inputs: BuildInputs) -> str:
    linked = link.write_week_links(week)
    return f"{week.name}: {linked} requests continue an earlier request"


def run_chains(week: paths.Week, inputs: BuildInputs) -> str:
    counts = chains.write_week_chains(week)
    return f"{week.name}: {counts['sessions']} sessions, {counts['agent_sessions']} of them agent sessions"


def run_spawns(week: paths.Week, inputs: BuildInputs) -> str:
    counts = spawns.write_week_spawns(week)
    return f"{week.name}: {counts['spawn_calls']} spawn calls, {counts['matched']} matched to a session"


def run_release_sessions(week: paths.Week, inputs: BuildInputs) -> str:
    counts = release_sessions.write_week_release_sessions(week)
    return f"{week.name}: {counts['population_sessions']} sessions the release is built from"


def run_system_prompts(week: paths.Week, inputs: BuildInputs) -> str:
    kept = system_prompts.write_week_system_prompts(week)
    return f"{week.name}: {kept} requests have a top-level system prompt"


def run_digests(week: paths.Week, inputs: BuildInputs) -> str:
    written = digests.write_week_digests(week, inputs.salt, inputs.tables.compaction_notes)
    return f"{week.name}: {written} digests"


def run_transitions(week: paths.Week, inputs: BuildInputs) -> str:
    counts = transitions.write_week_transitions(week)
    return f"{week.name}: transitions " + ", ".join(f"{count} {name}" for name, count in sorted(counts.items()))


def run_tool_calls(week: paths.Week, inputs: BuildInputs) -> str:
    written = calls.write_week_tool_calls(week, inputs.tables)
    return f"{week.name}: {written} tool calls"


# The steps in the order they run. `extract`, `system-prompts` and `digests` read the weekly log too, which
# `build_week` checks before the first step.
BUILD_STEPS = (
    Step("manifest", "the weekly log's manifest, when absent", (), (), run_manifest),
    Step("extract", "the week's requests that can belong to an agent session, with the rest counted",
         (), (paths.EXTRACT_FILE, paths.EXTRACT_COUNTS_FILE), run_extract),
    Step("signals", "the linking signals of each kept request", (paths.EXTRACT_FILE,), (paths.SIGNALS_FILE,),
         run_signals),
    Step("link", "each request's parent, the earlier request it continues", (paths.SIGNALS_FILE,),
         (paths.LINKS_FILE,), run_link),
    Step("chains", "each session's main chain and turns, and which sessions are agent sessions",
         (paths.SIGNALS_FILE, paths.LINKS_FILE), (paths.CHAINS_FILE,), run_chains),
    Step("spawns", "each spawn call matched to the session its subagent ran", (paths.SIGNALS_FILE, paths.CHAINS_FILE),
         (paths.SPAWN_MATCHES_FILE,), run_spawns),
    Step("release-sessions", "the requests, sessions and spawns the release is built from, and the week's counts",
         (paths.EXTRACT_FILE, paths.EXTRACT_COUNTS_FILE, paths.SIGNALS_FILE, paths.LINKS_FILE, paths.CHAINS_FILE,
          paths.SPAWN_MATCHES_FILE),
         (paths.REQUESTS_FILE, paths.SESSIONS_FILE, paths.SPAWNS_FILE, paths.COUNTS_FILE), run_release_sessions),
    Step("system-prompts", "the top-level system prompt of each kept request", (paths.EXTRACT_FILE,),
         (paths.SYSTEM_PROMPTS_FILE,), run_system_prompts),
    Step("digests", "each release request's message facts and block hashes",
         (paths.EXTRACT_FILE, paths.REQUESTS_FILE, paths.SYSTEM_PROMPTS_FILE), (paths.DIGESTS_FILE,), run_digests),
    Step("transitions", "each request's input against the step before: append, mutation or switch, and the cause",
         (paths.EXTRACT_FILE, paths.REQUESTS_FILE, paths.SYSTEM_PROMPTS_FILE, paths.DIGESTS_FILE),
         (paths.TRANSITIONS_FILE,), run_transitions),
    Step("tool-calls", "each tool call of a release request: as sent, and what is decided from it",
         (paths.EXTRACT_FILE, paths.REQUESTS_FILE), (paths.TOOL_CALLS_FILE, paths.TOOL_TURNS_FILE), run_tool_calls),
)
STEP_NAMES = tuple(step.name for step in BUILD_STEPS)

# The week files the release reads; a release stops when one is missing.
RELEASE_READS = (paths.EXTRACT_FILE, paths.REQUESTS_FILE, paths.SESSIONS_FILE, paths.DIGESTS_FILE,
                 paths.TRANSITIONS_FILE, paths.TOOL_CALLS_FILE)


def writer_of(file_name: str) -> str:
    """The step that writes a week file."""
    for step in BUILD_STEPS:
        if file_name in step.writes:
            return step.name
    raise ValueError(f"no step writes {file_name}")


def require_files(week: paths.Week, needed_by: str, file_names: tuple[str, ...]) -> None:
    """Stop unless every named file is in the week's folder; the message names the step that writes it."""
    for file_name in file_names:
        if not (week.folder / file_name).exists():
            raise MissingInputError(
                f"{week.name}: {needed_by} needs {file_name}; run "
                f"python -m export build --week {week.sunday} --from {writer_of(file_name)}"
            )


def build_week(week: paths.Week, inputs: BuildInputs, first_step: str = STEP_NAMES[0]) -> typing.Iterator[str]:
    """Run the steps from `first_step` to the last, yielding each step's sentence as it finishes."""
    if not week.log_file.exists():
        raise MissingInputError(
            f"{week.name}: no weekly log at {week.log_file}; link the week's log there, or run "
            f"python -m export cut-weeks --source FILE"
        )
    for step in BUILD_STEPS[STEP_NAMES.index(first_step):]:
        require_files(week, step.name, step.reads)
        yield step.run(week, inputs)
