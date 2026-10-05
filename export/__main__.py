"""The export's command line: `python -m export <command> ...`.

- `build --week W [--from STEP]`: a week's weekly log to its tool calls, every step in order (`build_steps.py`);
- `tool-words`: the public list of tool words, rebuilt over every week;
- `release --week W | --all-weeks`: a week's anonymous release, its converted tables, and `run.json`;
- `relabel --week W --harness H`: the harness of the week's sessions of that harness, decided again;
- `cut-weeks --source FILE`: the weeks that have no weekly log yet, cut out of a longer export;
- `report anonymisation | reference --reference DIR`: the two reports, which change nothing.
Every command prints one sentence per week it touched.
"""

import argparse
import dataclasses
import datetime
import pathlib

from release import convert
from export import paths, configs, settings, build_steps
from export.privacy import pseudonyms, public_words
from export.reports import anonymisation_report, reference_comparison
from export.log_records import log_files
from export.release import build, run_record
from export.sessions import relabel


@dataclasses.dataclass(frozen=True)
class Workspace:
    """What every command works in: the maintainer's settings, the data home and the rule tables."""

    export_settings: settings.ExportSettings
    data: pathlib.Path
    tables: configs.RuleTables

    def week(self, sunday: datetime.date) -> paths.Week:
        """One of the export's weeks, its folder made; stops for a Sunday that is not one."""
        if sunday not in self.export_settings.weeks:
            raise SystemExit(f"{sunday} is not a week of this export; see private/export.toml")
        week = paths.week(sunday, self.data, self.export_settings.max_processes)
        week.folder.mkdir(parents=True, exist_ok=True)
        return week


def week_folders_of(workspace: Workspace) -> list[pathlib.Path]:
    """The private folder of every week of the export."""
    return [paths.intermediate_week_directory(workspace.data, sunday) for sunday in workspace.export_settings.weeks]


def run_build(options: argparse.Namespace, workspace: Workspace) -> None:
    """Build one week from its weekly log, from the named step on, printing one line per step."""
    week = workspace.week(options.week)
    inputs = build_steps.BuildInputs(tables=workspace.tables, excluded_users=workspace.export_settings.excluded_users,
                                    salt=pseudonyms.load_salt(paths.SALT))
    for sentence in build_steps.build_week(week, inputs, options.from_step):
        print(sentence, flush=True)


def run_tool_words(options: argparse.Namespace, workspace: Workspace) -> None:
    """Rebuild the public list of tool words from every week, and say whether it changed."""
    before = configs.PUBLIC_TOOL_WORDS.read_bytes() if configs.PUBLIC_TOOL_WORDS.exists() else b""
    week_folders = week_folders_of(workspace)
    sizes = public_words.write_public_words(week_folders, configs.PUBLIC_TOOL_WORDS, workspace.tables.tool_tables)
    counts = ", ".join(f"{size} {name}" for name, size in sizes.items())
    if configs.PUBLIC_TOOL_WORDS.read_bytes() == before:
        print(f"the public tool words are unchanged: {counts}")
    else:
        print(f"the public tool words changed: {counts}; commit {configs.PUBLIC_TOOL_WORDS.name} and release "
              f"every week again (python -m export release --all-weeks)")


def run_release(options: argparse.Namespace, workspace: Workspace) -> None:
    """Release one week or every week: the trace lines and manifest, the converted tables, then `run.json`."""
    sundays = workspace.export_settings.weeks if options.all_weeks else (options.week,)
    keys = pseudonyms.load_private_keys(paths.SALT, paths.PROVIDERS)
    for sunday in sundays:
        week = workspace.week(sunday)
        build_steps.require_files(week, "release", build_steps.RELEASE_READS)
        week_manifest = build.write_release_week(week, workspace.tables, keys)
        tables_record = convert.convert_week(week.release_folder)
        traces = sum(entry["lines"] for entry in week_manifest["files"].values())
        rows = ", ".join(f"{facts['rows']} rows in {name}" for name, facts in tables_record["files"].items())
        print(f"{week.name}: {traces} traces in {len(week_manifest['files'])} trace files; {rows}", flush=True)
    record = run_record.write_run_record(workspace.export_settings, workspace.data)
    print(f"{paths.RUN_RECORD_FILE} records {len(record['weeks'])} weeks at code version {record['code_version']}")


def run_relabel(options: argparse.Namespace, workspace: Workspace) -> None:
    """Decide again the harness of the week's sessions of one harness, and say what they became."""
    week = workspace.week(options.week)
    build_steps.require_files(week, "relabel", (paths.EXTRACT_FILE, paths.SESSIONS_FILE))
    counts = relabel.relabel_week(week, options.harness, workspace.tables.harnesses)
    decided = ", ".join(f"{count} {name}" for name, count in sorted(counts.items())) or "none"
    print(f"{week.name}: {sum(counts.values())} sessions of {options.harness} decided again: {decided}")


def run_cut_weeks(options: argparse.Namespace, workspace: Workspace) -> None:
    """Cut the weeks that have no weekly log yet out of a longer export, and say what was read."""
    logs = paths.logs_directory(workspace.data)
    weeks = log_files.weeks_to_cut(workspace.export_settings.weeks, logs)
    if not weeks:
        print("every week has its weekly log; nothing to cut")
        return
    print(f"cutting {', '.join(paths.week_name(week) for week in weeks)} from {options.source}", flush=True)
    tally = log_files.cut_weeks(options.source, weeks, logs, paths.temporary_directory(workspace.data),
                                workspace.export_settings.max_processes)
    print(f"read {tally.lines} lines: {tally.unreadable_lines} unreadable, {tally.lines_of_other_weeks} of other weeks")


def run_report(options: argparse.Namespace, workspace: Workspace) -> None:
    """Write one of the two reports, and say where it is."""
    intermediate = paths.intermediate_directory(workspace.data)
    if options.kind == "reference":
        if options.reference is None:
            raise SystemExit("report reference needs --reference DIR, the reference export's folder")
        report = reference_comparison.write_comparison(workspace.export_settings, workspace.data, options.reference)
        same = report["total"]["percent"]["same_requests"]
        print(f"{same}% of the reference's {report['total']['reference_agent_sessions']} agent sessions have the same "
              f"requests here; see {intermediate / paths.REFERENCE_COMPARISON_FILE}")
        return
    week_folders = week_folders_of(workspace)
    output = intermediate / paths.ANONYMISATION_REPORT_FILE
    differing = anonymisation_report.write_report(week_folders, paths.release_directory(workspace.data),
                                                  pseudonyms.load_salt(paths.SALT), workspace.tables, output)
    names = ", ".join(sorted(differing)) or "none"
    print(f"{len(differing)} numbers differ after anonymisation: {names}; see {output}")


# Each command by its name on the command line.
COMMANDS = {
    "build": run_build,
    "tool-words": run_tool_words,
    "release": run_release,
    "relabel": run_relabel,
    "cut-weeks": run_cut_weeks,
    "report": run_report,
}
SUNDAY = {"type": datetime.date.fromisoformat, "metavar": "SUNDAY", "help": "the week's Sunday, YYYY-MM-DD"}


def parse_arguments(arguments: list[str] | None) -> argparse.Namespace:
    """The command and its options."""
    parser = argparse.ArgumentParser(prog="python -m export",
                                     description="The export: weekly weekly logs to the anonymous release.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    build_command = commands.add_parser("build", help="build a week from its weekly log, every step in order")
    build_command.add_argument("--week", required=True, **SUNDAY)
    build_command.add_argument("--from", dest="from_step", choices=build_steps.STEP_NAMES, metavar="STEP",
                               default=build_steps.STEP_NAMES[0],
                               help=f"the first step to run again; one of {', '.join(build_steps.STEP_NAMES)}")
    commands.add_parser("tool-words", help="rebuild the public list of tool words from every week")
    release_command = commands.add_parser("release",
                                          help="write a week's anonymous release and tables, and run.json")
    which = release_command.add_mutually_exclusive_group(required=True)
    which.add_argument("--week", **SUNDAY)
    which.add_argument("--all-weeks", action="store_true", help="every week of the export")
    relabel_command = commands.add_parser("relabel",
                                          help="decide again the harness of the week's sessions of one harness")
    relabel_command.add_argument("--week", required=True, **SUNDAY)
    relabel_command.add_argument("--harness", required=True, help="the harness name to decide again, such as pi")
    cut_command = commands.add_parser("cut-weeks",
                                      help="cut the weeks without a weekly log out of a longer export")
    cut_command.add_argument("--source", type=pathlib.Path, required=True, help="the longer export, as plain JSONL")
    report_command = commands.add_parser("report", help="write a report that changes nothing")
    report_command.add_argument("kind", choices=("anonymisation", "reference"),
                                help="anonymisation: every tool number before and after; "
                                     "reference: the export's sessions against a reference export's")
    report_command.add_argument("--reference", type=pathlib.Path, help="the reference export's folder")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> None:
    """Run one command in the maintainer's workspace."""
    options = parse_arguments(arguments)
    workspace = Workspace(export_settings=settings.load_export_settings(paths.EXPORT_SETTINGS),
                          data=paths.data_directory(), tables=configs.load())
    try:
        COMMANDS[options.command](options, workspace)
    except build_steps.MissingInputError as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
