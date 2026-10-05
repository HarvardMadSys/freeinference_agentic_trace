"""The prefix-cache simulation: `python -m simulation [--week SUNDAY]`.

One released week's cache replay and cheaper-reads model, or, without a week,
every week whose simulation is missing or was computed from other traces.
The release is read from `data/v1/release` and the simulation written to
`data/v1/simulation`, unless an option says otherwise.
"""

import argparse
import os
import pathlib

from release import reader
from simulation import cheaper_reads, simulate, weekly_files

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = REPOSITORY_ROOT / "data" / "v1" / "release"
DEFAULT_SIMULATION = REPOSITORY_ROOT / "data" / "v1" / "simulation"
MOST_DEFAULT_PROCESSES = 8  # partitions simulated at once unless --processes says otherwise: one needs a few GB on a
# large week, so the default stays within a reader's memory; a large machine passes --processes


def parse_arguments(arguments: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m simulation", description="Replay the prefix cache and the "
                                     "cheaper-reads model on one week, or on every week not simulated from its current traces.")
    parser.add_argument("--week", metavar="SUNDAY",
                        help="the week's Sunday, YYYY-MM-DD; without it, every week not simulated from its current traces")
    parser.add_argument("--release", type=pathlib.Path, default=DEFAULT_RELEASE, metavar="DIR", help="the release directory")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_SIMULATION, metavar="DIR", help="the simulation directory")
    parser.add_argument("--processes", type=int, default=min(MOST_DEFAULT_PROCESSES, os.cpu_count() or 1),
                        help="partitions simulated at once; the default is the smaller of 8 and the machine's cores")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> None:
    """Replay the cache and run the cheaper-reads model on one week, or on every week not simulated from its current
    traces."""
    options = parse_arguments(arguments)
    if options.week is not None:
        weeks = reader.week_directories(options.release, [options.week])
    else:
        weeks = weekly_files.stale_weeks(options.release, options.output)
        if not weeks:
            print("every released week is simulated from its current traces")
    for week_directory in weeks:
        simulate.simulate_week(week_directory, options.output, options.processes)
        written = cheaper_reads.write_week(week_directory, options.output)
        model = "the cheaper-reads model" if written is not None else "no cheaper-reads model (no Agentic requests)"
        print(f"{week_directory.name}: the cache replay and {model}, into {options.output / week_directory.name}",
              flush=True)


if __name__ == "__main__":
    main()
