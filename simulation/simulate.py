"""The cache simulation of one release week: every scope, policy and TTL, and the quantities that add across weeks.

Each scope (Agentic, and SWE-Bench on its own) is simulated with its cache
empty at the week's start. The week's partitions run in parallel and their
tallies are summed. Three files go to `data/v1/simulation/<week>/`:
- `tallies.parquet`: per scope, policy and TTL, hit and missed tokens, and
  reusable and dead token-seconds;
- `totals.parquet`: per scope, active token-seconds (blocks of requests in
  flight), the span in seconds, output tokens and requests;
- `request_misses.parquet`: per request, the tokens it missed with no TTL
  under `ttl_only`.
Its `sources.json` then records the sha256 of the week's manifest beside each
of them. Ratios, shares and quantiles are left to the figures, which combine weeks.
"""

import concurrent.futures
import multiprocessing
import pathlib

import numpy
import pandas

from simulation import block_reads, cache, weekly_files
from release import population

SCOPES = (population.AGENTIC, population.SWE_BENCH)


def simulate_partition(week_directory: pathlib.Path, week: block_reads.WeekRequests, partition: int, week_end_s: float) -> dict:
    """One partition's tallies per policy and TTL, and its reads and first reads per request."""
    hashes, request = block_reads.partition_reads(week_directory, week.in_scope, partition)
    sorted_reads = cache.sort_reads(hashes, request, week)
    tallies = {}
    for policy in cache.POLICIES:
        for ttl_name, ttl_s in cache.TTLS:
            tallies[(policy, ttl_name)] = cache.tally(sorted_reads, ttl_s, policy, week_end_s)
    requests = len(week.arrival_s)
    return {
        "tallies": tallies,
        "reads": numpy.bincount(sorted_reads.request, minlength=requests),
        "first_reads": numpy.bincount(sorted_reads.request[sorted_reads.first], minlength=requests),
    }


def simulate_scope(week_directory: pathlib.Path, scope: str, processes: int) -> tuple[list, dict, pandas.DataFrame] | None:
    """The scope's tally rows, its totals, and its per-request misses; None when the week has none of its requests."""
    week = block_reads.week_requests(week_directory, scope)
    if not week.in_scope.any():
        return None
    week_end_s = float(week.completion_s[week.in_scope].max())
    sums = {}
    reads_per_request = numpy.zeros(len(week.arrival_s), dtype=numpy.int64)
    first_reads = numpy.zeros(len(week.arrival_s), dtype=numpy.int64)
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=processes, mp_context=context) as pool:
        jobs = [pool.submit(simulate_partition, week_directory, week, partition, week_end_s)
                for partition in range(block_reads.PARTITIONS)]
        for job in jobs:
            result = job.result()
            reads_per_request += result["reads"]
            first_reads += result["first_reads"]
            for key, tally in result["tallies"].items():
                for name, value in tally.items():
                    sums.setdefault(key, {}).setdefault(name, 0)
                    sums[key][name] += value
    rows = [{"scope": scope, "policy": policy, "ttl": ttl_name, **sums[(policy, ttl_name)]} for policy, ttl_name in sums]
    return rows, scope_totals(week, scope, reads_per_request), request_misses(week, scope, first_reads)


def scope_totals(week: block_reads.WeekRequests, scope: str, reads_per_request: numpy.ndarray) -> dict:
    """The scope's active token-seconds, span, output tokens and requests."""
    in_scope = week.in_scope
    latency_s = week.completion_s - week.arrival_s
    return {
        "scope": scope,
        "active_token_seconds": float((cache.BLOCK_TOKENS * reads_per_request * latency_s)[in_scope].sum()),
        "span_seconds": float(week.completion_s[in_scope].max() - week.arrival_s[in_scope].min()),
        "output_tokens": float(week.output_tokens[in_scope].sum()),
        "requests": int(in_scope.sum()),
    }


def request_misses(week: block_reads.WeekRequests, scope: str, first_reads: numpy.ndarray) -> pandas.DataFrame:
    """Per in-scope request, the tokens it missed with no TTL under `ttl_only`: its blocks read first this week."""
    in_scope = week.in_scope
    return pandas.DataFrame({
        "scope": scope,
        "session_id": week.session_ids[in_scope],
        "step": week.step[in_scope],
        "missed_tokens": cache.BLOCK_TOKENS * first_reads[in_scope],
    })


def simulate_week(week_directory: pathlib.Path, simulation_directory: pathlib.Path, processes: int) -> None:
    """Simulate every scope of one release week, and write its three files."""
    tally_rows, totals, misses = [], [], []
    for scope in SCOPES:
        simulated = simulate_scope(week_directory, scope, processes)
        if simulated is None:
            continue
        rows, scope_total, scope_misses = simulated
        tally_rows.extend(rows)
        totals.append(scope_total)
        misses.append(scope_misses)
    output = simulation_directory / week_directory.name
    output.mkdir(parents=True, exist_ok=True)
    pandas.DataFrame(tally_rows).to_parquet(output / weekly_files.TALLIES_FILE, index=False)
    pandas.DataFrame(totals).to_parquet(output / weekly_files.TOTALS_FILE, index=False)
    pandas.concat(misses, ignore_index=True).to_parquet(output / weekly_files.REQUEST_MISSES_FILE, index=False)
    weekly_files.record_sources(output, week_directory, [weekly_files.TALLIES_FILE, weekly_files.TOTALS_FILE, weekly_files.REQUEST_MISSES_FILE])
