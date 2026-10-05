"""The cheaper-reads model on one release week: what today's harness pays, and what a harness that mutates less would pay.

Every step of the Agentic scope's sessions is charged under a TTL. A step that
arrives within the TTL of its previous step's completion is warm: it reads the
prompt tokens it shares with that step from cache and prefills the rest. A
cold step, and a session's first, prefills its whole prompt. A step's shared
share is the share of its blocks it has in common with the previous step, so
every charge is in the provider's tokens. A model switch is charged the same
way; the provider's reported cache reads are not used.

A reduction is a mutation that shrinks the prompt; its penalty is the prompt
tokens it prefills again when warm. At a read price cut d, the harness that
mutates less skips the d share of the week's reductions with the smallest
penalty. A skipped reduction keeps the tokens it would have removed in its
session's carry and adds nothing new; every later step reads the carry from
cache when warm and prefills it when cold, until the context grows past the
forced-compaction size, where the step prefills its own prompt and the carry
is dropped.
"""

import math
import pathlib

import numpy
import pandas

from release import schema
from release import population, reader
from simulation import weekly_files

KEYS = ["session_id", "step"]
STEP_COLUMNS = ["session_id", "step", "start_ms", "end_ms", "prompt_tokens", "completion_tokens", "transition"]
COMPARED_TTLS = (("5m", 300.0), ("15m", 900.0), ("none", math.inf))  # the TTLs the model is run under
PRICE_CUT_STEPS = 20  # the read price falls from $1 to free per million tokens in steps of 1/20
# A context past this many tokens, or past the session's largest prompt if larger, forces a compaction:
# 0.9 of a one-million-token window.
FORCED_COMPACTION_TOKENS = 900_000


class RequestFactMissing(Exception):
    """A week's requests lack a fact the model charges by."""


# ---- how many blocks a request's input shares with its session's previous step. Block hashes are chained, so two
# inputs share their first k blocks exactly when their first k hashes are equal; a session's first step shares none.
# requests.parquet is sorted by session and step, so a step's previous step is the row before it, which may sit in
# the row group before.


def common_prefix_length(first: numpy.ndarray, second: numpy.ndarray) -> int:
    """How many leading values the two arrays have in common."""
    length = min(len(first), len(second))
    differs = numpy.flatnonzero(first[:length] != second[:length])
    if len(differs):
        return int(differs[0])
    return length


def shared_prefixes(week_directory: pathlib.Path) -> pandas.DataFrame:
    """Per request: its session id and step, its blocks, and the blocks it shares with its session's previous step."""
    blocks = reader.open_blocks(week_directory)
    rows = {"session_id": [], "step": [], "blocks": [], "shared_blocks": []}
    previous_session = None
    previous_hashes = numpy.zeros(0, dtype=numpy.int64)
    for row_group in range(blocks.num_row_groups):
        table = blocks.read_row_group(row_group, columns=["session_id", "step", "block_ids"])
        lists = table["block_ids"].combine_chunks()
        values = lists.values.to_numpy()
        offsets = lists.offsets.to_numpy()
        for position, (session_id, step) in enumerate(zip(table["session_id"].to_pylist(), table["step"].to_pylist())):
            hashes = values[offsets[position]:offsets[position + 1]]
            shared = common_prefix_length(hashes, previous_hashes) if session_id == previous_session else 0
            rows["session_id"].append(session_id)
            rows["step"].append(step)
            rows["blocks"].append(len(hashes))
            rows["shared_blocks"].append(shared)
            previous_session, previous_hashes = session_id, hashes
    return pandas.DataFrame(rows)


def step_facts(requests: pandas.DataFrame, prefixes: pandas.DataFrame) -> pandas.DataFrame:
    """The requests in session and step order, each with its gap, its cached tokens when warm, and its reduction facts.

    `prefixes` is joined on session and step, whatever its order. A request
    without a row in `prefixes`, or without a prompt token count, stops the
    model.
    """
    steps = requests[STEP_COLUMNS].merge(prefixes[KEYS + ["blocks", "shared_blocks"]], on=KEYS, how="left",
                                         validate="one_to_one")
    if steps.blocks.isna().any():
        raise RequestFactMissing(f"{int(steps.blocks.isna().sum())} requests have no block ids")
    if steps.prompt_tokens.isna().any():
        raise RequestFactMissing(f"{int(steps.prompt_tokens.isna().sum())} requests have no prompt token count")
    steps = steps.sort_values(KEYS, ignore_index=True)
    session = steps.groupby("session_id")
    previous_end_ms = session.end_ms.shift()
    previous_prompt = session.prompt_tokens.shift()
    shared_share = (steps.shared_blocks / steps.blocks).where(steps.blocks > 0, 0.0)
    steps["first"] = previous_end_ms.isna()
    steps["gap_s"] = (steps.start_ms - previous_end_ms) / schema.MILLISECONDS_PER_SECOND
    steps["cached_if_warm"] = steps.prompt_tokens * shared_share
    steps["is_reduction"] = (steps.transition == "mutation") & (steps.prompt_tokens < previous_prompt)
    steps["penalty"] = (steps.prompt_tokens - steps.cached_if_warm).where(steps.is_reduction, 0.0)
    steps["removed_tokens"] = (previous_prompt - steps.prompt_tokens).where(steps.is_reduction, 0.0)
    steps["compaction_tokens"] = session.prompt_tokens.transform("max").clip(lower=FORCED_COMPACTION_TOKENS)
    return steps


def build_steps(week_directory: pathlib.Path) -> pandas.DataFrame | None:
    """The steps of the week's Agentic sessions, from its release files; None when it has none."""
    release = reader.load(week_directory.parent, [reader.week_sunday(week_directory)])
    requests = population.sessions_in_scope(release, population.AGENTIC).requests
    if requests.empty:
        return None
    prefixes = shared_prefixes(week_directory)
    return step_facts(requests, prefixes)


def warm_steps(steps: pandas.DataFrame, ttl_s: float) -> numpy.ndarray:
    """Whether each step arrives no more than the TTL after its previous step's completion; a first step never does."""
    return (steps.gap_s <= ttl_s).to_numpy()


def skipped_reductions(steps: pandas.DataFrame, cut_index: int) -> numpy.ndarray:
    """Whether each step is one of the reductions skipped at the price cut cut_index / PRICE_CUT_STEPS.

    They are the d × N reductions with the smallest penalty, N being the week's
    reductions, rounded half up; ties go to the earlier step in session and
    step order.
    """
    reductions = numpy.flatnonzero(steps.is_reduction.to_numpy())
    by_penalty = reductions[numpy.argsort(steps.penalty.to_numpy()[reductions], kind="stable")]
    count = math.floor(cut_index * len(reductions) / PRICE_CUT_STEPS + 0.5)
    skipped = numpy.zeros(len(steps), dtype=bool)
    skipped[by_penalty[:count]] = True
    return skipped


def charge(steps: pandas.DataFrame, warm: numpy.ndarray, skipped: numpy.ndarray) -> dict:
    """The prefilled and cached tokens of the steps, and the forced compactions, when the skipped reductions are not done.

    With nothing skipped, the carry stays 0 and this is today's harness.
    """
    prefilled = 0.0
    cached = 0.0
    forced_compactions = 0
    carry = 0.0
    columns = zip(steps["first"].tolist(), steps.prompt_tokens.tolist(), steps.cached_if_warm.tolist(),
                  steps.removed_tokens.tolist(), steps.compaction_tokens.tolist(), warm.tolist(), skipped.tolist())
    for first, prompt, cached_if_warm, removed, compaction_tokens, is_warm, is_skipped in columns:
        if first:
            carry = 0.0
        if is_skipped:
            carry += removed
        context = prompt + carry
        if context > compaction_tokens:
            prefilled += prompt
            forced_compactions += 1
            carry = 0.0
        elif is_skipped and is_warm:
            cached += context
        elif is_warm:
            prefilled += prompt - cached_if_warm
            cached += cached_if_warm + carry
        else:
            prefilled += context
    return {"prefilled_tokens": prefilled, "cached_tokens": cached, "forced_compactions": forced_compactions}


def week_rows(steps: pandas.DataFrame) -> list[dict]:
    """One row per TTL and price cut: today's and the harness that mutates less's token sums, and the counts."""
    nothing_skipped = numpy.zeros(len(steps), dtype=bool)
    output_tokens = float(steps.completion_tokens.fillna(0).sum())
    reductions = int(steps.is_reduction.sum())
    rows = []
    for ttl_name, ttl_s in COMPARED_TTLS:
        warm = warm_steps(steps, ttl_s)
        today = charge(steps, warm, nothing_skipped)
        for cut_index in range(PRICE_CUT_STEPS + 1):
            skipped = skipped_reductions(steps, cut_index)
            mutates_less = charge(steps, warm, skipped)
            rows.append({
                "ttl": ttl_name,
                "read_price_cut": cut_index / PRICE_CUT_STEPS,
                "today_prefilled_tokens": today["prefilled_tokens"],
                "today_cached_tokens": today["cached_tokens"],
                "prefilled_tokens": mutates_less["prefilled_tokens"],
                "cached_tokens": mutates_less["cached_tokens"],
                "output_tokens": output_tokens,
                "reductions": reductions,
                "skipped_reductions": int(skipped.sum()),
                "forced_compactions": mutates_less["forced_compactions"],
            })
    return rows


def write_week(week_directory: pathlib.Path, simulation_directory: pathlib.Path) -> pathlib.Path | None:
    """Write the week's `cheaper_reads.parquet` into its simulation folder, and record its source; None, and no file,
    without Agentic requests."""
    steps = build_steps(week_directory)
    output = simulation_directory / week_directory.name
    output.mkdir(parents=True, exist_ok=True)
    path = output / weekly_files.CHEAPER_READS_FILE
    if steps is not None:
        pandas.DataFrame(week_rows(steps)).to_parquet(path, index=False)
    weekly_files.record_sources(output, week_directory, [weekly_files.CHEAPER_READS_FILE])
    return path if steps is not None else None
