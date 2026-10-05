"""The paper's tables and figures, drawn from the release in the paper's style, and the registry that says which is which.

One package per section of the paper, each with a `draw` that writes the
section's folder of the output: `dataset`, `sessions`, `tools`,
`prefix_caching` and `benchmark_vs_production`. `inputs` is what every section
is drawn from, and `plot_style` holds how every figure looks. `FIGURES` below is the only place
in the code that names a paper figure or table; `index_text` writes the output
folder's README, which names the files and what they show.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Item:
    """One table or figure of the paper: what the paper calls it, and where it comes from."""

    paper: str  # what the paper calls it, such as "Figure 8b"
    section: str  # the folder of the output it is written to, and its section in numbers.json
    key: str  # its key in numbers.json, under the section
    function: str  # the function that draws it, under paper
    outputs: tuple[str, ...]  # the file stems it writes in the section's folder; empty when it is numbers only
    shows: str  # one sentence: what the figure or table shows


FIGURES = (
    Item("Table 1", "dataset", "trace_summary", "dataset.trace_summary.write_trace_summary",
         ("trace_summary", "trace_summary_by_harness"),
         "The trace's totals (sessions, requests, tool calls, tokens) and sessions and input tokens per harness"),
    Item("Figure 2", "sessions", "counts_per_session", "sessions.structure.counts_per_session", ("counts_per_session",),
         "CDFs of requests, user steps and tool steps per session"),
    Item("Figure 2", "sessions", "turn_and_session_duration", "sessions.structure.turn_and_session_duration",
         ("turn_and_session_duration",),
         "CDFs of how long one user turn lasts and how long a whole session lasts"),
    Item("Figure 2", "sessions", "tokens_per_request", "sessions.tokens.tokens_per_request", ("tokens_per_request",),
         "CDFs of input, uncached input and output tokens per request"),
    Item("Figure 2", "sessions", "cost_share_by_session", "sessions.tokens.cost_share_by_session",
         ("cost_share_by_session",),
         "Each session's cost split between cached input, uncached input and output"),
    Item("Figure 3", "sessions", "token_growth_by_step", "sessions.tokens.token_growth", ("token_growth_by_step",),
         "Cached, uncached and output tokens as the number of messages in the request grows"),
    Item("Figure 3", "sessions", "context_share_by_role", "sessions.tokens.context_share_by_role",
         ("context_share_by_role",),
         "The system, user, assistant and tool shares of the input as it grows"),
    Item("Figure 3", "sessions", "context_share_by_role_with_definitions", "sessions.tokens.context_share_with_definitions",
         ("context_share_by_role_with_definitions",),
         "The same shares with the tool definitions counted beside the system prompt"),
    Item("Figure 4", "sessions", "ttft_decomposition_minimax-m2.7", "sessions.latency.ttft_decomposition",
         ("ttft_decomposition_minimax-m2.7",),
         "CDFs of TTFT, decode time and queueing for minimax-m2.7"),
    Item("Figure 4", "sessions", "ttft_decomposition_kimi-k2.7-code", "sessions.latency.ttft_decomposition",
         ("ttft_decomposition_kimi-k2.7-code",),
         "CDFs of TTFT, decode time and queueing for kimi-k2.7-code"),
    Item("Figure 4", "sessions", "session_time_breakdown", "sessions.latency.session_time_breakdown",
         ("session_time_breakdown",),
         "Each session's time split between prefill, decode, tools and the person"),
    Item("Figure 4", "sessions", "speedup_if_twice_as_fast", "sessions.latency.speedup_if_twice_as_fast",
         ("speedup_if_twice_as_fast",),
         "The session-time speedup if prefill, decode or tools alone ran twice as fast"),
    Item("text", "tools", "shell_split", "tools.overview.shell_split", (),
         "The share of tool calls that are shell commands"),
    Item("Table 4", "tools", "example_commands", "tools.overview.example_commands", ("example_commands",),
         "The most frequent programs of each shell category"),
    Item("Table 5", "tools", "tool_overview", "tools.overview.overview_table", ("tool_overview",),
         "Per tool category: shares of calls, sessions, result tokens and wait, failure rate and batching"),
    Item("Figure 5a", "tools", "latency_by_category", "tools.overview.latency_and_result_size",
         ("latency_by_category",),
         "Per tool category, the wait a call causes"),
    Item("Figure 5b", "tools", "result_tokens_by_category", "tools.overview.latency_and_result_size",
         ("result_tokens_by_category",),
         "Per tool category, the size of a call's result"),
    Item("Table 6a", "tools", "batching_profile", "tools.batching.batching_profile", ("batching_profile",),
         "How often calls are batched: sessions with a batch, calls per request, and the batch mix"),
    Item("Figure 6a", "tools", "latency_by_batch_width", "tools.batching.latency_by_width", ("latency_by_batch_width",),
         "The wait of a batch of calls by its width, against issuing them one per request"),
    Item("Figure 6b", "tools", "calls_per_request_by_category", "tools.batching.calls_per_request",
         ("calls_per_request_by_category",),
         "The mean number of calls per request, by the category of the first call"),
    Item("Figure 8", "prefix_caching", "hit_ratio_by_gap_bin", "prefix_caching.reuse.hit_ratio_by_gap_bin",
         ("hit_ratio_by_gap_bin",),
         "CDFs of the cache hit ratio by how long the request waited"),
    Item("Figure 8b", "prefix_caching", "retention_by_provider", "prefix_caching.reuse.retention_by_provider",
         ("retention_by_provider",),
         "The median hit ratio against the idle gap at each provider"),
    Item("Figure 8", "prefix_caching", "gap_by_cause", "prefix_caching.reuse.gap_by_cause", ("gap_by_cause",),
         "CDFs of the gap before a request by what it waited for"),
    Item("Figure 8", "prefix_caching", "mean_hit_ratio_by_cause", "prefix_caching.reuse.mean_hit_ratio_by_cause",
         ("mean_hit_ratio_by_cause",),
         "The mean hit ratio by what the request waited for"),
    Item("Table 7", "prefix_caching", "mutations_by_cause", "prefix_caching.context_mutations.mutation_table",
         ("mutations_by_cause",),
         "Context mutations by cause: their shares of steps, sessions and prefill"),
    Item("Figure 9", "prefix_caching", "hit_ratio_after_transition", "prefix_caching.context_mutations.hit_ratio_after_transition",
         ("hit_ratio_after_transition",),
         "CDFs of the hit ratio after an append and after a mutation"),
    Item("Figure 10a", "prefix_caching", "prefill_against_storage", "prefix_caching.cache_retention.prefill_against_storage",
         ("prefill_against_storage",),
         "Prefill against retained tokens, one point per cache TTL"),
    Item("Figure 10b", "prefix_caching", "retained_over_active", "prefix_caching.cache_retention.retained_over_active",
         ("retained_over_active",),
         "Reusable and dead retained tokens over active ones, per TTL"),
    Item("Figure 11a", "prefix_caching", "api_cost", "prefix_caching.cache_retention.api_cost", ("api_cost",),
         "API cost per TTL, and the TTL where it is lowest"),
    Item("Figure 11b", "prefix_caching", "infrastructure_cost", "prefix_caching.cache_retention.infrastructure_cost",
         ("infrastructure_cost",),
         "Recompute and storage cost per TTL, and the TTL where their total is lowest"),
    Item("Figure 12", "prefix_caching", "cheaper_reads_prefill", "prefix_caching.cheaper_reads.draw",
         ("cheaper_reads_prefill",),
         "Prefill with today's harness against one that mutates less, as cache reads get cheaper"),
    Item("Figure 12", "prefix_caching", "cheaper_reads_cost", "prefix_caching.cheaper_reads.draw", ("cheaper_reads_cost",),
         "API cost with today's harness against one that mutates less, as cache reads get cheaper"),
    Item("Figure 13a", "prefix_caching", "dead_over_active_by_policy", "prefix_caching.cache_retention.by_policy",
         ("dead_over_active_by_policy",),
         "Dead over active tokens per TTL, one line per eviction policy"),
    Item("Figure 13b", "prefix_caching", "infrastructure_cost_by_policy", "prefix_caching.cache_retention.by_policy",
         ("infrastructure_cost_by_policy",),
         "Recompute and storage cost per TTL, one line per eviction policy"),
    Item("Figure 14a", "benchmark_vs_production", "turn_duration", "benchmark_vs_production.comparison.turn_duration",
         ("turn_duration",),
         "CDFs of one user turn's duration, human-driven against benchmark"),
    Item("Figure 14b", "benchmark_vs_production", "session_duration", "benchmark_vs_production.comparison.session_duration",
         ("session_duration",),
         "CDFs of a session's duration, human-driven against benchmark"),
    Item("Figure 14c", "benchmark_vs_production", "input_per_request", "benchmark_vs_production.comparison.input_per_request",
         ("input_per_request",),
         "CDFs of the input tokens of one request, human-driven against benchmark"),
    Item("Figure 14d", "benchmark_vs_production", "user_prompt_size", "benchmark_vs_production.comparison.user_prompt_size",
         ("user_prompt_size",),
         "CDFs of the size of a user's new prompt, human-driven against benchmark"),
    Item("Figure 14e", "benchmark_vs_production", "prefill_against_storage", "benchmark_vs_production.prefill_against_storage",
         ("prefill_against_storage",),
         "Prefill against retained tokens per TTL, human-driven against benchmark"),
    Item("Figure 15", "benchmark_vs_production", "trajectory_mix_human", "benchmark_vs_production.trajectories.trajectory_mix",
         ("trajectory_mix_human",),
         "Which kinds of tool calls happen at each tenth of a turn, human-driven"),
    Item("Figure 15", "benchmark_vs_production", "trajectory_mix_benchmark", "benchmark_vs_production.trajectories.trajectory_mix",
         ("trajectory_mix_benchmark",),
         "Which kinds of tool calls happen at each tenth of a turn, benchmark"),
)

def items_of(paper: str) -> list[Item]:
    """The registry's items the paper calls `paper`, such as "Figure 8b"; the whole figure for a figure number alone."""
    wanted = paper.strip().lower()
    return [item for item in FIGURES if item.paper.lower() == wanted or item.paper.lower().rstrip("abcde") == wanted]


def index_text() -> str:
    """The output folder's README: per section, each file and what it shows."""
    lines = ["# The paper's figures", "",
             "Drawn by `python -m paper` from the release in `data/v1`, one folder per section of the paper; each figure is "
             "a PDF and a PNG, each table also a CSV or JSON.", ""]
    for section in dict.fromkeys(item.section for item in FIGURES):
        lines += [f"## `{section}/`", "", "| file | shows |", "|---|---|"]
        for item in FIGURES:
            if item.section == section and item.outputs:
                lines.append(f"| {', '.join(f'`{stem}`' for stem in item.outputs)} | {item.shows} |")
        lines.append("")
    return "\n".join(lines)
