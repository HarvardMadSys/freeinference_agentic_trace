"""The sessions section, over the Agentic scope: session structure, tokens and context, and latency."""

import pathlib

from paper import inputs
from paper.sessions import latency, structure, tokens
from release import population


def draw(figure_inputs: inputs.Inputs, section_directory: pathlib.Path) -> dict:
    """Draw the section's figures, and return their numbers beside the paper's."""
    agentic = population.sessions_in_scope(figure_inputs.release, population.AGENTIC)
    numbers = {
        "counts_per_session": structure.counts_per_session(agentic, section_directory),
        "turn_and_session_duration": structure.turn_and_session_duration(agentic, section_directory),
        "tokens_per_request": tokens.tokens_per_request(agentic, section_directory),
        "cost_share_by_session": tokens.cost_share_by_session(agentic, section_directory),
        "token_growth_by_step": tokens.token_growth(agentic, section_directory),
        "context_share_by_role": tokens.context_share_by_role(agentic, section_directory),
        "context_share_by_role_with_definitions": tokens.context_share_with_definitions(agentic, section_directory),
        "session_time_breakdown": latency.session_time_breakdown(agentic, section_directory),
        "speedup_if_twice_as_fast": latency.speedup_if_twice_as_fast(agentic, section_directory),
    }
    for model in latency.MODELS:
        numbers[f"ttft_decomposition_{model}"] = latency.ttft_decomposition(agentic, section_directory, model)
    for figure_numbers in numbers.values():
        figure_numbers["scope"] = population.AGENTIC
    return numbers
