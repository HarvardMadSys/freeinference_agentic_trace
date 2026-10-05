"""Before and after anonymisation: every tool number computed from the full calls and from the anonymised ones.

The private tool tables hold each call in full (`raw_name`, `programs_full`,
`arguments_skeleton_full`); the report anonymises them with the committed
public word list, as the release does (`name`, `programs`,
`arguments_skeleton`). It puts each version under the release's column names,
computes the tool figures' numbers on both with the same code the figures use,
and writes them side by side to `tool_anonymisation.json` in the intermediate
folder, with the share of calls whose name ships as `other`, and of calls
whose program, key or command word was replaced. It reads the release for the
requests and each call's latency, so it runs after the release.
"""

import json
import pathlib

import pandas
from pyarrow import parquet

from release import population, reader
from release import schema
from paper.metrics import tools, trajectories
from export import paths, configs
from export.privacy import pseudonyms, tool_columns
from export.tool_calls import arguments

FULL_COLUMNS = {"raw_name": "name", "programs_full": "programs", "arguments_skeleton_full": "arguments_skeleton"}
ANONYMISED_COLUMNS = ("name", "programs", "arguments_skeleton")
# What the report reads of a private call: the released columns, the full ones, and the arguments the subagent type
# is read from.
PRIVATE_COLUMNS = ["session_id", "step", "call_index", "category", "parse_status", "outcome", "result_tokens",
                   "raw_name", "programs_full", "arguments_skeleton_full", "arguments"]


def private_calls(week_folder: pathlib.Path, salt: bytes, tables: configs.RuleTables) -> pandas.DataFrame:
    """The week's private calls in full and anonymised with the committed list, keyed by the release's hashed session
    id."""
    full = parquet.read_table(week_folder / paths.TOOL_CALLS_FILE, columns=PRIVATE_COLUMNS)
    for name, column in tool_columns.anonymised_columns(full, tables.public_words, tables.tool_tables).items():
        full = full.append_column(name, column)
    calls = full.drop_columns(["arguments"]).to_pandas()
    calls["session_id"] = [pseudonyms.session_hash(session_id, salt) for session_id in calls.session_id]
    return calls


def release_shaped(calls: pandas.DataFrame, full: bool) -> pandas.DataFrame:
    """The calls under the release's column names: the full columns when `full`, else the anonymised ones."""
    if full:
        chosen = calls.drop(columns=list(ANONYMISED_COLUMNS)).rename(columns=FULL_COLUMNS)
    else:
        chosen = calls.drop(columns=list(FULL_COLUMNS))
    return chosen[list(schema.TOOL_CALLS_SCHEMA.names)]


def flat(prefix: str, value: object, into: dict) -> None:
    """Every leaf of a nested number under a dotted name."""
    if isinstance(value, dict):
        for key, inner in value.items():
            flat(f"{prefix}.{key}", inner, into)
    else:
        into[prefix] = value


def tool_numbers(tool_calls: pandas.DataFrame, release: reader.Release) -> dict:
    """Every number of the tool figures, as dotted names."""
    agentic = population.sessions_in_scope(release, population.AGENTIC)
    benchmark = population.sessions_in_scope(release, population.SWE_BENCH)
    calls = tools.calls_frame(tool_calls, agentic.requests)
    numbers = {}
    flat("shell_share", tools.shell_share(calls), numbers)
    flat("tool_overview", tools.overview_table(calls, len(agentic.sessions)).round(3).to_dict(orient="index"), numbers)
    flat("example_commands", {category: ", ".join(labels) for category, labels in tools.example_commands(calls).items()},
         numbers)
    flat("batching_profile", tools.batching_profile(calls, len(agentic.sessions)), numbers)
    for name, scope in (("human", agentic), ("benchmark", benchmark)):
        shares = trajectories.trajectory_shares(tool_calls, scope.requests).round(4)
        flat(f"trajectory_mix_{name}", {str(tenth): row.to_dict() for tenth, row in shares.iterrows()}, numbers)
    return numbers


def collapse_shares(calls: pandas.DataFrame) -> dict:
    """The share of calls whose name ships as `other`, and of calls whose program, key or command word the public
    list replaced."""
    replaced_program = [full is not None and list(full) != list(public)
                        for full, public in zip(calls.programs_full, calls.programs)]
    replaced_command = [arguments.command_values(full) != arguments.command_values(public)
                        for full, public in zip(calls.arguments_skeleton_full, calls.arguments_skeleton)]
    return {
        "name": float((calls.name == schema.OTHER_WORD).mean()),
        "program": float(pandas.Series(replaced_program).mean()),
        "key": float(calls.arguments_skeleton.str.contains(f'"{tool_columns.KEY_MARK}"', regex=False).mean()),
        "command_word": float(pandas.Series(replaced_command).mean()),
    }


def write_report(week_folders: list[pathlib.Path], release_directory: pathlib.Path, salt: bytes,
                 tables: configs.RuleTables, output: pathlib.Path) -> dict:
    """Write the before-and-after report over every week, and return its differing numbers."""
    calls = pandas.concat([private_calls(week, salt, tables) for week in week_folders], ignore_index=True)
    release = reader.load(release_directory)
    latencies = release.tool_calls[["session_id", "step", "call_index", "latency_ms"]]
    calls = calls.merge(latencies, on=["session_id", "step", "call_index"], how="left")
    full = tool_numbers(release_shaped(calls, full=True), release)
    anonymised = tool_numbers(release_shaped(calls, full=False), release)
    differing = {name: {"full": full[name], "anonymised": anonymised.get(name)}
                 for name in full if full[name] != anonymised.get(name)}
    report = {"collapsed": collapse_shares(calls), "differing": differing,
              "numbers": {name: {"full": full[name], "anonymised": anonymised.get(name)} for name in full}}
    output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return differing
