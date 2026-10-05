<h1 align="center">FreeInference Agentic Trace</h1>

<p align="center">
  <strong>From Requests to Sessions: A Large-Scale Characterization of Human-Driven Agentic Workloads</strong>
  <br>
  <sub>16 weeks · 12K agent sessions · 209B input tokens · 1.2M requests · 1.2M tool calls</sub>
</p>

<p align="center">
  <a href="#dataset">Dataset</a>
  ·
  <a href="#reproducing-the-paper">Reproduction</a>
  ·
  <a href="#citation">Citation</a>
</p>

---

This repository contains the dataset and reproducibility artifacts for
**From Requests to Sessions: A Large-Scale Characterization of Human-Driven Agentic Workloads**.

We analyze 16 weeks (2026-05-17 to 2026-09-05) of coding and assistant agents using
[FreeInference](https://freeinference.org/), covering **12,002 agent sessions**
from **267 accounts**, spanning **209B input tokens**, **1,186,582 LLM requests**,
and **1,213,347 tool calls**. 

The trace captures rich metadata about agentic sessions, including request timing
and latency, models and anonymized providers, token usage, tool definitions and
calls, and block-level prefix hashes for replaying KV-cache reuse.

**No message text, prompt text, tool results, or raw identifiers are released.**

## Dataset

The dataset is hosted on Hugging Face:

[harvardMadsys/freeinference_agentic_trace](https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace)

We provide two representations:

- **Raw JSONL traces** — the original nested session representation, about **108 GB decompressed**.
- **Parquet tables** — flattened session, request, and tool-call tables used by the paper, about **9 GB**.

The raw traces preserve the complete session structure. The Parquet tables are
more convenient for analysis and are sufficient for reproducing the paper's
figures.

### Trace format

The trace is organized by agent session. Each session contains an ordered sequence
of LLM requests and their tool calls. Sessions created by subagents are nested
under the tool call that spawned them.

Each line of `traces/<day>.jsonl` contains one top-level session.

<details>
<summary><strong>A simplified example of one trace line</strong></summary>

```jsonc
{
  "week": "2026-05-17_2026-05-23",
  "day": "2026-05-17",

  // Session field
  "session_id": "...",
  "user_id": "...",
  "harness": "claude-code",
  "start_ms": 1778977459236,
  "end_ms": 1778984067198,

  // LLM requests in chronological order.
  "requests": [
    {
      // Position in the session and user turn.
      "step": 0,
      "turn_index": 0,

      // What triggered the request and how it connects
      // to the previous request.
      "step_trigger": "user",
      "link_reason": null,

      // Request timing at the gateway.
      "start_ms": 1778977459236,
      "end_ms": 1778977468828,
      "ttft_ms": 7889,

      // Requested model and anonymized upstream provider.
      "model": "glm-5.1",
      "provider": "B",
      "stream": true,

      // Token counts reported by the provider.
      "provider_tokens": {
        "prompt": 52731,
        "completion": 230,
        "cached": 1280
      },

      // Input tokens broken down by message role, tokenized with tiktoken.
      "role_tokens": {
        "system": 1790,
        "user": 1110,
        "assistant": 15916,
        "tool": 25725,
        "tool_definitions": 8638
      },

      // Shape of the input and tools exposed to the model.
      "n_messages": 179,
      "tool_definition_names": [
        "read_file",
        "write_file",
        "terminal",
        "execute_code"
      ],
      "finish_reason": "tool_calls",

      // How the input differs from the previous request.
      // Null for the first request in a session.
      "mutation": null,

      // Input represented as chained 16-token blocks,
      // tokenized with tiktoken.
      "block_ids": [123, 456, 789, "..."],

      // Tool calls produced by the model response.
      "tool_calls": [
        {
          "call_index": 0,
          "name": "terminal",

          // Normalized tool type. Shell calls additionally record
          // the programs found in the command.
          "category": "*Interpreter",
          "programs": ["python"],
          "parse_status": "success",

          // Arguments preserve their structure while sensitive
          // values are replaced with typed placeholders.
          "arguments": [
            [
              "command",
              "python <path> --status"
            ],
            [
              "timeout",
              "<num>"
            ]
          ],

          // Tool outcome and the gap until the next LLM request.
          "outcome": "error",
          "result_tokens": 842,
          "latency_ms": 5272,

          // If this call starts another agent, its complete
          // session is nested here.
          "subagent": null
        }
      ]
    },

    {
      "step": 1,
      "turn_index": 0,
      "step_trigger": "tool",
      "link_reason": "tool_call_id",

      // Later requests contain the same fields.
      // Here the previous input was extended without changing
      // any earlier content.
      "mutation": {
        "transition": "append",
        "cause": null
      },

      "tool_calls": ["..."]
    }
  ]
}
```

</details>

### Download

Set up the environment:

```bash
make setup
```

Download the raw JSONL traces:

```bash
.venv/bin/python -m release download --traces
```

Convert the downloaded traces into Parquet tables:

```bash
.venv/bin/python -m release convert
```

Or download the pre-generated Parquet tables directly:

```bash
.venv/bin/python -m release download --tables
```

## Reproducing the paper

The Parquet tables are sufficient for reproducing the paper's analysis.

```bash
make figures
```

This generates the paper's tables and figures under `figures/v1/`.

The prefix-cache simulation results used by the paper are also included in the
repository. To recompute one week:

```bash
python -m simulation --week 2026-08-30
```

## Repository structure

```text
release/      dataset format, reader, downloader, and conversion
simulation/   prefix-cache replay and cost simulation
paper/        analysis and figure code
export/       release and anonymization pipeline
figures/v1/   reproduced paper figures and tables
data/v1/      downloaded release data and simulation outputs
```

## License

The dataset and this artifact are released under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

You may share and adapt the material for any purpose, including commercially,
provided you give appropriate credit — please cite the paper below.

## Citation

If you use the dataset or artifact in your research, please cite our paper:

```bibtex
@article{freeinference2026requests,
  title={From Requests to Sessions: A Large-Scale Characterization of Human-Driven Agentic Workloads},
  author={Nixon, William and Tian, Muxin and Zheng, Yunjia and Gunawi, Haryadi S. and Yang, Juncheng},
  year={2026}
}
```