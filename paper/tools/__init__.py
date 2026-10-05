"""The tools section, over the Agentic scope: which tools agents call, what each costs, and how they batch.

It reads the release's tool calls with their requests.
"""

import pathlib

from paper import inputs
from paper.tools import batching, overview
from release import population
from paper.metrics import tools


def draw(figure_inputs: inputs.Inputs, section_directory: pathlib.Path) -> dict:
    """Draw the section's tables and figures, and return their numbers beside the paper's."""
    agentic = population.sessions_in_scope(figure_inputs.release, population.AGENTIC)
    calls = tools.calls_frame(agentic.tool_calls, agentic.requests)
    session_count = len(agentic.sessions)
    numbers = {
        "shell_split": overview.shell_split(calls),
        "example_commands": overview.example_commands(calls, section_directory),
        "tool_overview": overview.overview_table(calls, session_count, section_directory),
        "batching_profile": batching.batching_profile(calls, session_count, section_directory),
        "latency_by_batch_width": batching.latency_by_width(calls, section_directory),
        "calls_per_request_by_category": batching.calls_per_request(calls, section_directory),
    }
    numbers.update(overview.latency_and_result_size(calls, section_directory))
    numbers["shell_split"]["tool_calls"] = int(len(calls))  # the trace summary counts every scope
    for figure_numbers in numbers.values():
        figure_numbers["scope"] = population.AGENTIC
    return numbers
