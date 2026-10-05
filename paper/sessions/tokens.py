"""Tokens: per request, their cost per session, their growth along the context, and each role's share."""

import pathlib

import pandas

from release import reader
from paper.plot_style import axes, marks, style
from paper.metrics import binning, tokens

MIN_TOKENS_DRAWN = 10  # token axes start at 10
MAX_TOKENS_DRAWN = 1_000_000


def tokens_per_request(population: reader.Release, output_directory: pathlib.Path) -> dict:
    """CDFs of input, uncached input and output tokens per request."""
    requests = population.requests
    series = {
        "input": ("Input", requests.prompt_tokens.dropna(), (-12, 27), "right"),
        "uncached": ("Input (Uncached)", tokens.uncached_tokens(requests).dropna(), (12, -27), "left"),
        "output": ("Output", requests.completion_tokens[requests.completion_tokens > 1], (-12, 27), "right"),
    }
    figure, plot = style.new_figure(top_legend=True)
    numbers = {}
    for name, (label, values, offset, align) in series.items():
        color = style.TOKEN_MIX_COLOR[label]
        marks.cdf_line(plot, values, label, color, floor=MIN_TOKENS_DRAWN)
        median = binning.median(values)
        marks.median_label(plot, median, color, marks.count_text(median), offset, align)
        numbers[f"median_{name}"] = median
    axes.apply(plot, "x", "tokens", "tokens", low=MIN_TOKENS_DRAWN, high=MAX_TOKENS_DRAWN)
    axes.apply(plot, "y", "cdf", "requests")
    marks.legend_above(plot)
    style.save(figure, output_directory, "tokens_per_request")
    return numbers


def cost_share_by_session(population: reader.Release, output_directory: pathlib.Path) -> dict:
    """Each session's cost split between cached input, uncached input and output, in rank bins."""
    shares = tokens.cost_shares_per_session(population.requests)
    binned = binning.rank_bins(shares, ["cached", "uncached"])
    bands = [("Input (Cached)", binned.cached, style.TOKEN_SHARE_COLOR["Input (Cached)"]),
             ("Input (Uncached)", binned.uncached, style.TOKEN_SHARE_COLOR["Input (Uncached)"]),
             ("Output", binned.output, style.TOKEN_SHARE_COLOR["Output"])]
    figure, plot = style.new_figure(top_legend=True)
    marks.stack_area(plot, binned.percentile, bands)
    axes.apply(plot, "x", "percentile", low=0, high=100)
    axes.apply(plot, "y", "share", "cost", low=0, high=1)
    plot.grid(False, axis="x", which="both")
    marks.legend_above(plot)
    style.save(figure, output_directory, "cost_share_by_session")
    pooled = tokens.pooled_cost_shares(population.requests)
    return {f"pooled_{name}": share for name, share in pooled.items()}


def token_growth(population: reader.Release, output_directory: pathlib.Path) -> dict:
    """Cached, uncached (not clipped) and output tokens, as size-binned profiles along the number of messages in the
    request."""
    rows = tokens.growth_rows(population.requests)
    values = pandas.DataFrame({"Input (Cached)": rows.cache_read_tokens,
                               "Input (Uncached)": rows.prompt_tokens - rows.cache_read_tokens,
                               "Output": rows.completion_tokens})
    table = binning.size_bins(rows.n_messages, values)
    window = (1.0, float(table.mid.max()))
    figure, plot = style.new_figure(top_legend=True)
    for label in values.columns:
        marks.band_line(plot, table[table.series == label], label, style.GROWTH_COLOR[label], window,
                        band_floor=MIN_TOKENS_DRAWN)
    axes.apply(plot, "x", "count", "messages in request", *window)
    axes.apply(plot, "y", "tokens", "tokens", low=MIN_TOKENS_DRAWN, high=MAX_TOKENS_DRAWN)
    marks.legend_above(plot)
    style.save(figure, output_directory, "token_growth_by_step")
    return {"negative_uncached_share": float((values["Input (Uncached)"] < 0).mean())}


def context_share(population: reader.Release, output_directory: pathlib.Path, parts: dict, name: str) -> dict:
    """Each part's share of the input, as size-binned profiles along the number of messages in the request."""
    requests = population.requests[population.requests.n_messages >= 1]
    shares = tokens.role_shares(requests, parts)
    table = binning.size_bins(requests.loc[shares.index, "n_messages"], shares)
    window = (1.0, float(table.mid.max()))
    figure, plot = style.new_figure(top_legend=True)
    for label in shares.columns:
        marks.band_line(plot, table[table.series == label], label, style.ROLE_COLOR[label], window)
    axes.apply(plot, "x", "count", "messages in request", *window)
    axes.apply(plot, "y", "share", "context", low=0, high=1)
    marks.legend_above(plot)
    style.save(figure, output_directory, name)
    return {"requests": int(len(shares))}


def context_share_by_role(population: reader.Release, output_directory: pathlib.Path) -> dict:
    """The four message roles' shares of the messages' tokens."""
    return context_share(population, output_directory, tokens.MESSAGE_ROLES, "context_share_by_role")


def context_share_with_definitions(population: reader.Release, output_directory: pathlib.Path) -> dict:
    """The same, with the tool definitions counted beside the system prompt, over the whole input."""
    name = "context_share_by_role_with_definitions"
    return context_share(population, output_directory, tokens.DEFINITIONS_AS_SYSTEM, name)
