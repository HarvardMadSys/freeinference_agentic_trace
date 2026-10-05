"""The prefix-caching section, over the Agentic scope.

Its gap and hit-ratio figures need only the requests. The mutation table and
the hit ratio after a transition need the cache simulation's misses, the
retention figures its tallies, and the cheaper-reads figure its model; when
those are missing, the figures are not drawn and `numbers.json` says so.
"""

import pathlib

from release import population, reader
from paper import inputs
from paper.plot_style import style
from paper.prefix_caching import cache_retention, cheaper_reads, context_mutations, reuse
from simulation import cache, weekly_files
from paper.metrics import cache_costs, mutations


def mutation_figures(agentic: reader.Release, figure_inputs: inputs.Inputs, section_directory: pathlib.Path) -> dict:
    """The mutation table and the hit ratio after a transition, from the requests' transitions and each request's misses with no TTL."""
    misses = weekly_files.request_misses(figure_inputs.simulation_directory, figure_inputs.release_directory)
    frame = mutations.transition_rows(agentic.requests, misses[misses.scope == population.AGENTIC])
    return {
        "mutations_by_cause": context_mutations.mutation_table(frame, section_directory),
        "hit_ratio_after_transition": context_mutations.hit_ratio_after_transition(frame, section_directory),
    }


def retention_figures(figure_inputs: inputs.Inputs, section_directory: pathlib.Path) -> dict:
    """The retention and cost figures, from the simulation's tallies summed over the weeks."""
    tallies = weekly_files.tallies(figure_inputs.simulation_directory, figure_inputs.release_directory)
    totals = weekly_files.totals(figure_inputs.simulation_directory, figure_inputs.release_directory)
    curves = {policy: cache_costs.curve(tallies, totals, population.AGENTIC, policy) for policy in cache.POLICIES}
    ttl_only = curves["ttl_only"]
    cache_retention.prefill_against_storage({"Human-driven": (ttl_only, style.BLUE, cache_retention.HUMAN_TTL_LABELS)},
                                            section_directory, "prefill_against_storage")
    cache_retention.retained_over_active(ttl_only, section_directory)
    highest_cost = max(cache_retention.drawn_values(curves, "infrastructure_total")) * 1.1
    dead_window = cache_retention.decade_window(cache_retention.drawn_values(curves, "dead_over_active"))
    cache_retention.by_policy(curves, "dead_over_active", section_directory, "dead_over_active_by_policy", "count",
                        "dead / active", dead_window)
    cache_retention.by_policy(curves, "infrastructure_total", section_directory, "infrastructure_cost_by_policy",
                        "whole_multiple", "normalized cost", (0, highest_cost))
    numbers = cache_costs.retention_summary(curves, cache_retention.PREFILL_TTLS, cache_retention.RETAINED_TTLS)
    numbers["infrastructure_cost_by_policy"] = {}
    numbers["api_cost"] = cache_retention.api_cost(ttl_only, section_directory)
    numbers["infrastructure_cost"] = cache_retention.infrastructure_cost(ttl_only, section_directory)
    return numbers


# The figures `retention_figures` draws, and the two panels of the cheaper-reads figure, each a key of numbers.json.
RETENTION_FIGURES = ("prefill_against_storage", "retained_over_active", "dead_over_active_by_policy",
                     "infrastructure_cost_by_policy", "api_cost", "infrastructure_cost")
CHEAPER_READS_FIGURES = ("cheaper_reads_prefill", "cheaper_reads_cost")


def not_drawn(reason: str) -> dict:
    """The numbers of a figure that could not be drawn: why."""
    return {"drawn": f"no: {reason}"}


def draw(figure_inputs: inputs.Inputs, section_directory: pathlib.Path) -> dict:
    """Draw the section's figures, and return their numbers beside the paper's."""
    agentic = population.sessions_in_scope(figure_inputs.release, population.AGENTIC)
    numbers = {
        "hit_ratio_by_gap_bin": reuse.hit_ratio_by_gap_bin(agentic, section_directory),
        "retention_by_provider": reuse.retention_by_provider(agentic, section_directory),
        "gap_by_cause": reuse.gap_by_cause(agentic, section_directory),
        "mean_hit_ratio_by_cause": reuse.mean_hit_ratio_by_cause(agentic, section_directory),
    }
    has_simulation = weekly_files.has_weeks(figure_inputs.simulation_directory)
    if has_simulation:
        numbers.update(mutation_figures(agentic, figure_inputs, section_directory))
    else:
        numbers["mutations_by_cause"] = not_drawn("no week was simulated")
    if has_simulation:
        numbers.update(retention_figures(figure_inputs, section_directory))
    else:
        numbers.update({name: not_drawn("no week was simulated") for name in RETENTION_FIGURES})
    if weekly_files.has_cheaper_reads(figure_inputs.simulation_directory):
        sums = weekly_files.cheaper_reads(figure_inputs.simulation_directory, figure_inputs.release_directory)
        curves = cache_costs.cheaper_reads_curves(sums)
        numbers.update(cheaper_reads.draw(curves, section_directory))
    else:
        numbers.update({name: not_drawn("no week's cheaper-reads model was simulated") for name in CHEAPER_READS_FIGURES})
    for figure_numbers in numbers.values():
        figure_numbers["scope"] = population.AGENTIC
    return numbers
