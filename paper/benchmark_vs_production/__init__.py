"""The benchmark-against-production section, Agentic against SWE-Bench.

The prefill-against-storage figure needs the cache simulation's tallies; without them it is not drawn,
and `numbers.json` says so.
"""

import pathlib

from paper import inputs
from paper.benchmark_vs_production import comparison, trajectories
from paper.plot_style import style
from paper.prefix_caching import cache_retention
from simulation import weekly_files
from release import population
from paper.metrics import cache_costs


def prefill_against_storage(figure_inputs: inputs.Inputs, section_directory: pathlib.Path) -> dict:
    """Each scope's prefill against storage, each simulated on its own."""
    tallies = weekly_files.tallies(figure_inputs.simulation_directory, figure_inputs.release_directory)
    totals = weekly_files.totals(figure_inputs.simulation_directory, figure_inputs.release_directory)
    human = cache_costs.curve(tallies, totals, population.AGENTIC, "ttl_only")
    benchmark = cache_costs.curve(tallies, totals, population.SWE_BENCH, "ttl_only")
    curves = {"Human-driven": (human, style.SCOPE_COLOR["Human-driven"], cache_retention.HUMAN_TTL_LABELS_BESIDE_BENCHMARK),
              "Benchmark": (benchmark, style.SCOPE_COLOR["Benchmark"], cache_retention.BENCHMARK_TTL_LABELS)}
    cache_retention.prefill_against_storage(curves, section_directory, "prefill_against_storage")
    return {
        "benchmark_1m_above_floor": float(benchmark.normalised_prefill["1m"] - 1),
        "benchmark_5m_above_floor": float(benchmark.normalised_prefill["5m"] - 1),
        "production_15m_above_floor": float(human.normalised_prefill["15m"] - 1),
    }


def draw(figure_inputs: inputs.Inputs, section_directory: pathlib.Path) -> dict:
    """Draw the section's figures, and return their numbers beside the paper's."""
    human = population.sessions_in_scope(figure_inputs.release, population.AGENTIC)
    benchmark = population.sessions_in_scope(figure_inputs.release, population.SWE_BENCH)
    numbers = {
        "turn_duration": comparison.turn_duration(human, benchmark, section_directory),
        "session_duration": comparison.session_duration(human, benchmark, section_directory),
        "input_per_request": comparison.input_per_request(human, benchmark, section_directory),
        "user_prompt_size": comparison.user_prompt_size(human, benchmark, section_directory),
    }
    tool_calls = figure_inputs.release.tool_calls
    numbers["trajectory_mix_benchmark"] = trajectories.trajectory_mix(tool_calls, benchmark, section_directory,
                                                                      "trajectory_mix_benchmark")
    numbers["trajectory_mix_human"] = trajectories.trajectory_mix(tool_calls, human, section_directory,
                                                                  "trajectory_mix_human")
    benchmark_simulated = False
    if weekly_files.has_weeks(figure_inputs.simulation_directory):
        totals = weekly_files.totals(figure_inputs.simulation_directory, figure_inputs.release_directory)
        benchmark_simulated = population.SWE_BENCH in totals.index
    if benchmark_simulated:
        numbers["prefill_against_storage"] = prefill_against_storage(figure_inputs, section_directory)
    else:
        numbers["prefill_against_storage"] = {"drawn": "no: no SWE-Bench week was simulated"}
    for figure_numbers in numbers.values():
        figure_numbers["scope"] = f"{population.AGENTIC} and {population.SWE_BENCH}"
    return numbers
