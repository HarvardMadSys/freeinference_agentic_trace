"""The digests step: each release request reduced to what the release keeps.

A request's input is its tool definitions, system prompt and messages as sent
(`log_records/messages.py`), each entry rendered as `log_records/rendering.py` renders
it. Per entry, the digest keeps its role, its o200k tokens, its salted hash
and whether it holds a compaction note; the tool definitions never do. Per
request, it keeps the chained hashes of its 16-token blocks and the names of
the tools it declared. Later steps read the digests, never the bodies.

The bodies are read once, from `extract.parquet`, by row groups in several
processes. Each process writes its rows to a part file, since a request's
block hashes run to tens of thousands; the parts are joined in order into
`digests.parquet`.
"""

import dataclasses
import hashlib
import pathlib
import shutil

import numpy
import pyarrow
import tiktoken
from pyarrow import compute, parquet

from release import schema
from export import parallel, paths
from export.log_records import messages, rendering



HASH_BYTES = 8  # a salted hash is 64 bits, so two blocks of a week never share one by chance
ROWS_PER_WRITE = 100  # requests held in memory before a part file is written to

DIGESTS_SCHEMA = pyarrow.schema(
    [
        ("request_id", pyarrow.string()),
        ("message_roles", pyarrow.list_(pyarrow.string())),
        ("message_tokens", pyarrow.list_(pyarrow.int32())),
        ("message_hashes", pyarrow.list_(pyarrow.uint64())),
        ("compaction_notes", pyarrow.list_(pyarrow.bool_())),
        ("block_hashes", pyarrow.list_(pyarrow.uint64())),
        ("tool_definition_names", pyarrow.list_(pyarrow.string())),  # one per declared tool; null when it has no name
    ]
)


def salted_hash(data: bytes, salt: bytes) -> int:
    """The salted 8-byte BLAKE2b of `data`, as an unsigned integer."""
    digest = hashlib.blake2b(data, digest_size=HASH_BYTES, key=salt).digest()
    return int.from_bytes(digest, "little")


def block_hashes(tokens: list[int], salt: bytes) -> list[int]:
    """The chained hashes of the input's full 16-token blocks.

    A block is its tokens as 4-byte little-endian integers. The first block's
    hash is of its tokens; every later block's is of the previous hash followed
    by its tokens, so two inputs share a hash exactly when they share every
    token up to the end of that block.
    """
    token_bytes = numpy.asarray(tokens, dtype="<u4").tobytes()
    block_bytes = schema.BLOCK_TOKENS * 4
    hashes = []
    previous = b""
    for start in range(0, len(token_bytes) - block_bytes + 1, block_bytes):
        previous = hashlib.blake2b(previous + token_bytes[start:start + block_bytes],
                                   digest_size=HASH_BYTES, key=salt).digest()
        hashes.append(int.from_bytes(previous, "little"))
    return hashes


@dataclasses.dataclass(frozen=True)
class RequestInput:
    """What a request sent the model, as the extract and the system prompts hold it."""

    request_id: str
    prompt: str | None  # the record's `prompt`: the messages, as JSON text
    system_prompt: str | None  # the payload's top-level system prompt, when the harness sends one
    tools: str | None  # the record's `tools`: the tool definitions, as JSON text


def request_digest(request: RequestInput, encoding, salt: bytes, notes: tuple[str, ...]) -> dict:
    """The digest of one request."""
    digest = {"request_id": request.request_id, "message_roles": [], "message_tokens": [], "message_hashes": [],
              "compaction_notes": []}
    all_tokens = []
    for message in messages.input_messages(request.prompt, request.system_prompt, request.tools):
        text = rendering.render_message(message)
        tokens = encoding.encode_ordinary(text)
        role = rendering.role_of(message)
        digest["message_roles"].append(role)
        digest["message_tokens"].append(len(tokens))
        digest["message_hashes"].append(salted_hash(text.encode(), salt))
        digest["compaction_notes"].append(role != messages.TOOL_DEFINITIONS and any(note in text for note in notes))
        all_tokens.extend(tokens)
    digest["block_hashes"] = block_hashes(all_tokens, salt)
    digest["tool_definition_names"] = [messages.tool_name(tool) for tool in messages.declared_tools(request.tools)]
    return digest


def system_prompts_of(system_prompts_file: pathlib.Path, request_ids: pyarrow.Array) -> dict[str, str]:
    """The top-level system prompts of the given requests, by request id."""
    table = parquet.read_table(system_prompts_file, columns=["request_id", "system"])
    table = table.filter(compute.is_in(table["request_id"], value_set=request_ids))
    return dict(zip(table["request_id"].to_pylist(), table["system"].to_pylist()))


def request_ids_of(extract: parquet.ParquetFile, first: int, stop: int) -> pyarrow.Array:
    """The request ids of row groups `first` up to `stop` of the extract."""
    ids = [extract.read_row_group(row_group, columns=["request_id"])["request_id"] for row_group in range(first, stop)]
    return pyarrow.chunked_array(ids, type=pyarrow.string()).combine_chunks()


def digest_row_groups(shard: parallel.Shard, wanted: pyarrow.Array, salt: bytes, notes: tuple[str, ...]) -> int:
    """Write the digests of the wanted requests in the shard's row groups of the extract to its part file."""
    encoding = tiktoken.get_encoding(schema.TOKENIZER)
    extract = parquet.ParquetFile(shard.file)
    system_prompts = system_prompts_of(shard.file.parent / paths.SYSTEM_PROMPTS_FILE, request_ids_of(extract, shard.first, shard.stop))
    written = 0
    with parquet.ParquetWriter(shard.part_file, DIGESTS_SCHEMA, compression="none") as writer:
        rows = []
        for row_group in range(shard.first, shard.stop):
            table = extract.read_row_group(row_group, columns=["request_id", "prompt", "tools"])
            table = table.filter(compute.is_in(table["request_id"], value_set=wanted))
            for request in table.to_pylist():
                request_input = RequestInput(request["request_id"], request["prompt"], system_prompts.get(request["request_id"]),
                                             request["tools"])
                rows.append(request_digest(request_input, encoding, salt, notes))
                if len(rows) == ROWS_PER_WRITE:
                    writer.write_table(pyarrow.Table.from_pylist(rows, schema=DIGESTS_SCHEMA))
                    written += len(rows)
                    rows = []
        if rows:
            writer.write_table(pyarrow.Table.from_pylist(rows, schema=DIGESTS_SCHEMA))
            written += len(rows)
    return written


def write_week_digests(week: paths.Week, salt: bytes, compaction_notes: tuple[str, ...]) -> int:
    """Write the week's `digests.parquet` for its release requests, and return how many."""
    request_ids = parquet.read_table(week.folder / paths.REQUESTS_FILE, columns=["request_id"])["request_id"]
    wanted = pyarrow.array(sorted(request_ids.to_pylist()), type=pyarrow.string())
    parts_directory = parallel.parts_directory(week.temporary_folder, "digest", week.name)
    shards = parallel.row_group_shards(week.folder / paths.EXTRACT_FILE, parts_directory, week.processes)
    with parallel.process_pool(week.processes) as pool:
        jobs = [pool.submit(digest_row_groups, shard, wanted, salt, compaction_notes) for shard in shards]
        written = sum(job.result() for job in jobs)
    if written != len(wanted):
        raise ValueError(f"{week.name}: {written} digests for {len(wanted)} release requests")
    parallel.join_parts([shard.part_file for shard in shards], week.folder / paths.DIGESTS_FILE, DIGESTS_SCHEMA)
    shutil.rmtree(parts_directory)
    return written
