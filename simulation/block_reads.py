"""The block reads of one release week, for one scope.

Every block of every request's input is one read, tied to its request by the
request's row in `requests.parquet`; the requests are put in that file's order,
by session and step, before anything is read. Reads are split by
block hash into partitions: two partitions share no block, and a block's
history depends on no other block, so each partition is simulated on its own.
"""

import dataclasses
import pathlib

import numpy
import pandas

from release import schema
from release import population, reader

PARTITIONS = 64  # a large week's two billion reads, in parts of about thirty million


@dataclasses.dataclass(frozen=True)
class WeekRequests:
    """The week's requests, in the order of `requests.parquet`; per request, the arrays below."""

    arrival_s: numpy.ndarray
    completion_s: numpy.ndarray
    session: numpy.ndarray  # the session's index, the same for every request of a session
    step: numpy.ndarray
    output_tokens: numpy.ndarray
    in_scope: numpy.ndarray  # whether the request belongs to the scope simulated
    session_ids: numpy.ndarray  # the release's session id, to key the per-request rows written
    next_step_arrival_s: numpy.ndarray  # the arrival of the session's next step; infinite for its last
    session_end_s: numpy.ndarray  # the last completion of the request's session


def in_blocks_order(requests: pandas.DataFrame, week_directory: pathlib.Path) -> pandas.DataFrame:
    """The requests in the order of the week's `requests.parquet` file, matched on session and step.

    A read is tied to its request by its row's position in the file, so
    the requests must be in that file's order, whatever order they were read in.
    """
    keys = reader.open_blocks(week_directory).read(columns=["session_id", "step"]).to_pandas()
    indexed = requests.set_index(["session_id", "step"])
    missing = pandas.MultiIndex.from_frame(keys).difference(indexed.index)
    if len(missing) or len(keys) != len(requests):
        raise ValueError(f"{week_directory.name}: the loaded requests are not those of requests.parquet")
    return indexed.loc[pandas.MultiIndex.from_frame(keys)].reset_index()


def week_requests(week_directory: pathlib.Path, scope: str) -> WeekRequests:
    """The week's requests, in the order of its `requests.parquet`, marked by whether they are the scope's."""
    release = reader.load(week_directory.parent, [reader.week_sunday(week_directory)])
    requests = in_blocks_order(release.requests, week_directory)
    sessions = population.sessions_in_scope(release, scope).sessions
    session_index = pandas.factorize(requests.session_id)[0]
    arrival_s = requests.start_ms.to_numpy(dtype=float) / schema.MILLISECONDS_PER_SECOND
    completion_s = requests.end_ms.to_numpy(dtype=float) / schema.MILLISECONDS_PER_SECOND
    same_session_next = numpy.append(session_index[1:] == session_index[:-1], False)
    next_step_arrival_s = numpy.where(same_session_next, numpy.append(arrival_s[1:], numpy.inf), numpy.inf)
    session_end_s = pandas.Series(completion_s).groupby(session_index).transform("max").to_numpy()
    return WeekRequests(
        arrival_s=arrival_s,
        completion_s=completion_s,
        session=session_index,
        step=requests.step.to_numpy(),
        output_tokens=requests.completion_tokens.fillna(0).to_numpy(dtype=float),
        in_scope=requests.session_id.isin(sessions.session_id).to_numpy(),
        session_ids=requests.session_id.to_numpy(),
        next_step_arrival_s=next_step_arrival_s,
        session_end_s=session_end_s,
    )


def partition_reads(week_directory: pathlib.Path, in_scope: numpy.ndarray, partition: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    """The partition's reads of in-scope requests: their block hashes, and their requests' positions."""
    blocks = reader.open_blocks(week_directory)
    hashes, positions = [], []
    first_request = 0
    for row_group in range(blocks.num_row_groups):
        lists = blocks.read_row_group(row_group, columns=["block_ids"])["block_ids"].combine_chunks()
        values = lists.values.to_numpy()
        lengths = numpy.diff(lists.offsets.to_numpy())
        request = numpy.repeat(numpy.arange(first_request, first_request + len(lengths), dtype=numpy.int32), lengths)
        kept = (values % PARTITIONS == partition) & in_scope[request]
        hashes.append(values[kept])
        positions.append(request[kept])
        first_request += len(lengths)
    return numpy.concatenate(hashes), numpy.concatenate(positions)
