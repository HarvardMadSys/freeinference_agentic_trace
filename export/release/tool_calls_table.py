"""The release's `tool_calls.parquet`: the week's private tool calls anonymised, keyed by session and step.

The private table (`data/export/intermediate/<week>/tool_calls.parquet`) holds
every call in full. The release anonymises its name, programs and argument
skeleton with the public word list it is released under, reads only the
columns it needs (so an anonymised column an earlier build left in the file
is ignored), and adds each call's latency: the wait after its request until
the next, which every call of a batch shares.
"""

import pathlib

import pyarrow
from pyarrow import parquet

from release import schema
from export.privacy import public_words, tool_columns
from export.tool_calls import categories

COPIED_COLUMNS = ["call_index", "category", "parse_status", "outcome", "result_tokens"]  # released as they are
FULL_COLUMNS = ["raw_name", "programs_full", "arguments_skeleton_full", "arguments"]  # anonymised before release


def tool_calls_table(tool_calls_file: pathlib.Path, keys: dict[str, schema.RequestKey],
                     latencies: dict[schema.RequestKey, int], words: public_words.PublicWords,
                     tool_tables: categories.ToolTables) -> pyarrow.Table:
    """The released requests' calls, keyed by hashed session id and step, in session, step and call order.

    `keys` gives a released request's hashed session id and step by its raw id;
    `latencies` the wait after a request, by its session and step; `words` the
    public word list the calls are anonymised with.
    """
    private = parquet.read_table(tool_calls_file, columns=["request_id"] + COPIED_COLUMNS + FULL_COLUMNS)
    kept = pyarrow.array([request_id in keys for request_id in private["request_id"].to_pylist()])
    private = private.filter(kept)
    request_keys = [keys[request_id] for request_id in private["request_id"].to_pylist()]
    columns = {
        "session_id": pyarrow.array([key.session_id for key in request_keys], pyarrow.string()),
        "step": pyarrow.array([key.step for key in request_keys], pyarrow.int32()),
    }
    for name in COPIED_COLUMNS:
        columns[name] = private[name]
    columns.update(tool_columns.anonymised_columns(private, words, tool_tables))
    columns["arguments_skeleton"] = columns["arguments_skeleton"].cast(pyarrow.string())
    columns["latency_ms"] = pyarrow.array([latencies.get(key) for key in request_keys], pyarrow.int64())
    table = pyarrow.table(columns).select(schema.TOOL_CALLS_SCHEMA.names).cast(schema.TOOL_CALLS_SCHEMA)
    return table.sort_by([("session_id", "ascending"), ("step", "ascending"), ("call_index", "ascending")])
