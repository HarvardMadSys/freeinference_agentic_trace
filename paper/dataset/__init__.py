"""The dataset section: the trace summary, over every scope."""

import pathlib

from paper import inputs
from paper.dataset import trace_summary
from release import population


def draw(figure_inputs: inputs.Inputs, section_directory: pathlib.Path) -> dict:
    """Write the trace summary, and return its numbers beside the paper's."""
    trace_numbers = trace_summary.write_trace_summary(figure_inputs.release, section_directory)
    return {"trace_summary": {"scope": population.ALL_SCOPES, **trace_numbers}}
