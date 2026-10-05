# The paper's figures

Drawn by `python -m paper` from the release in `data/v1`, one folder per section of the paper; each figure is a PDF and a PNG, each table also a CSV or JSON.

## `dataset/`

| file | shows |
|---|---|
| `trace_summary`, `trace_summary_by_harness` | The trace's totals (sessions, requests, tool calls, tokens) and sessions and input tokens per harness |

## `sessions/`

| file | shows |
|---|---|
| `counts_per_session` | CDFs of requests, user steps and tool steps per session |
| `turn_and_session_duration` | CDFs of how long one user turn lasts and how long a whole session lasts |
| `tokens_per_request` | CDFs of input, uncached input and output tokens per request |
| `cost_share_by_session` | Each session's cost split between cached input, uncached input and output |
| `token_growth_by_step` | Cached, uncached and output tokens as the number of messages in the request grows |
| `context_share_by_role` | The system, user, assistant and tool shares of the input as it grows |
| `context_share_by_role_with_definitions` | The same shares with the tool definitions counted beside the system prompt |
| `ttft_decomposition_minimax-m2.7` | CDFs of TTFT, decode time and queueing for minimax-m2.7 |
| `ttft_decomposition_kimi-k2.7-code` | CDFs of TTFT, decode time and queueing for kimi-k2.7-code |
| `session_time_breakdown` | Each session's time split between prefill, decode, tools and the person |
| `speedup_if_twice_as_fast` | The session-time speedup if prefill, decode or tools alone ran twice as fast |

## `tools/`

| file | shows |
|---|---|
| `example_commands` | The most frequent programs of each shell category |
| `tool_overview` | Per tool category: shares of calls, sessions, result tokens and wait, failure rate and batching |
| `latency_by_category` | Per tool category, the wait a call causes |
| `result_tokens_by_category` | Per tool category, the size of a call's result |
| `batching_profile` | How often calls are batched: sessions with a batch, calls per request, and the batch mix |
| `latency_by_batch_width` | The wait of a batch of calls by its width, against issuing them one per request |
| `calls_per_request_by_category` | The mean number of calls per request, by the category of the first call |

## `prefix_caching/`

| file | shows |
|---|---|
| `hit_ratio_by_gap_bin` | CDFs of the cache hit ratio by how long the request waited |
| `retention_by_provider` | The median hit ratio against the idle gap at each provider |
| `gap_by_cause` | CDFs of the gap before a request by what it waited for |
| `mean_hit_ratio_by_cause` | The mean hit ratio by what the request waited for |
| `mutations_by_cause` | Context mutations by cause: their shares of steps, sessions and prefill |
| `hit_ratio_after_transition` | CDFs of the hit ratio after an append and after a mutation |
| `prefill_against_storage` | Prefill against retained tokens, one point per cache TTL |
| `retained_over_active` | Reusable and dead retained tokens over active ones, per TTL |
| `api_cost` | API cost per TTL, and the TTL where it is lowest |
| `infrastructure_cost` | Recompute and storage cost per TTL, and the TTL where their total is lowest |
| `cheaper_reads_prefill` | Prefill with today's harness against one that mutates less, as cache reads get cheaper |
| `cheaper_reads_cost` | API cost with today's harness against one that mutates less, as cache reads get cheaper |
| `dead_over_active_by_policy` | Dead over active tokens per TTL, one line per eviction policy |
| `infrastructure_cost_by_policy` | Recompute and storage cost per TTL, one line per eviction policy |

## `benchmark_vs_production/`

| file | shows |
|---|---|
| `turn_duration` | CDFs of one user turn's duration, human-driven against benchmark |
| `session_duration` | CDFs of a session's duration, human-driven against benchmark |
| `input_per_request` | CDFs of the input tokens of one request, human-driven against benchmark |
| `user_prompt_size` | CDFs of the size of a user's new prompt, human-driven against benchmark |
| `prefill_against_storage` | Prefill against retained tokens per TTL, human-driven against benchmark |
| `trajectory_mix_human` | Which kinds of tool calls happen at each tenth of a turn, human-driven |
| `trajectory_mix_benchmark` | Which kinds of tool calls happen at each tenth of a turn, benchmark |
