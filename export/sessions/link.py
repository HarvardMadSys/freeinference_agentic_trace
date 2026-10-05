"""Link: each request's parent, the earlier request of the same account it continues.

Requests are linked per account, in order of `end_ms` then `request_id`. A
request's parent is the nearest of the account's previous 128 requests that
either emitted a tool-call id this request replays (a tool link), or whose
answer phrases reappear, at least 9 in 10, in this request's last assistant
message (a text link). A tool link anywhere in the 128 beats a text link. A
request without a parent starts a session.
"""

import collections
import dataclasses
import pathlib

import pyarrow
from pyarrow import parquet

from export import parallel, paths

LOOKBACK_REQUESTS = 128  # a request may continue one of the account's previous 128 kept requests
TEXT_LINK_NUMERATOR = 9  # a text link needs at least 9/10 of the parent's answer phrases
TEXT_LINK_DENOMINATOR = 10  # to reappear in the child's last assistant message


LINKS_SCHEMA = pyarrow.schema(
    [
        ("request_id", pyarrow.string()),
        ("parent_request_id", pyarrow.string()),
        ("link", pyarrow.string()),
    ]
)

SIGNAL_COLUMNS = ["request_id", "user_id", "end_ms", "emitted_ids", "replayed_ids", "answer_shingles", "replayed_shingles"]


class DuplicateRequestIds(Exception):
    """Two kept requests of a week share a `request_id`."""


@dataclasses.dataclass(frozen=True)
class Request:
    """What linking reads of one request."""

    request_id: str
    user_id: str | None
    end_ms: int
    emitted_ids: frozenset[str]
    replayed_ids: frozenset[str]
    answer_shingles: frozenset[int]
    replayed_shingles: frozenset[int]


def is_tool_link(candidate: Request, request: Request) -> bool:
    """The candidate emitted a tool-call id the request replays."""
    return not candidate.emitted_ids.isdisjoint(request.replayed_ids)


def is_text_link(candidate: Request, request: Request) -> bool:
    """At least 9 in 10 of the candidate's answer phrases are in the request's last assistant message."""
    if not candidate.answer_shingles or not request.replayed_shingles:
        return False
    shared = len(candidate.answer_shingles & request.replayed_shingles)
    return TEXT_LINK_DENOMINATOR * shared >= TEXT_LINK_NUMERATOR * len(candidate.answer_shingles)


def parent_of(request: Request, candidates: list[Request]) -> tuple[str | None, str | None]:
    """The parent's `request_id` and the link kind, or `(None, None)`; candidates are nearest first."""
    for candidate in candidates:
        if is_tool_link(candidate, request):
            return candidate.request_id, "tool_id"
    for candidate in candidates:
        if is_text_link(candidate, request):
            return candidate.request_id, "text"
    return None, None


def link_account(account_requests: list[Request]) -> list[dict]:
    """The links of one account's requests, which are in order of `end_ms` then `request_id`."""
    links = []
    for position, request in enumerate(account_requests):
        window_start = max(0, position - LOOKBACK_REQUESTS)
        candidates = list(reversed(account_requests[window_start:position]))
        parent_request_id, link = parent_of(request, candidates)
        links.append({"request_id": request.request_id, "parent_request_id": parent_request_id, "link": link})
    return links


def link_requests(requests: list[Request]) -> list[dict]:
    """The links of a week's requests, in order of `end_ms` then `request_id`."""
    check_unique_request_ids(requests)
    ordered = sorted(requests, key=lambda request: (request.end_ms, request.request_id))
    by_account = collections.defaultdict(list)
    for request in ordered:
        by_account[request.user_id].append(request)
    link_by_request = {}
    for account_requests in by_account.values():
        for link in link_account(account_requests):
            link_by_request[link["request_id"]] = link
    return [link_by_request[request.request_id] for request in ordered]


def check_unique_request_ids(requests: list[Request]) -> None:
    """Stop with an error naming the `request_id`s that occur more than once."""
    counts = collections.Counter(request.request_id for request in requests)
    duplicates = sorted(request_id for request_id, count in counts.items() if count > 1)
    if duplicates:
        raise DuplicateRequestIds(f"{len(duplicates)} request ids occur more than once, such as {duplicates[:5]}")


def read_requests(signals_file: pathlib.Path) -> list[Request]:
    """The week's requests, as linking reads them, from its `signals.parquet`."""
    requests = []
    for row in parquet.read_table(signals_file, columns=SIGNAL_COLUMNS).to_pylist():
        requests.append(
            Request(
                request_id=row["request_id"],
                user_id=row["user_id"],
                end_ms=row["end_ms"],
                emitted_ids=frozenset(row["emitted_ids"]),
                replayed_ids=frozenset(row["replayed_ids"]),
                answer_shingles=frozenset(row["answer_shingles"]),
                replayed_shingles=frozenset(row["replayed_shingles"]),
            )
        )
    return requests


def write_week_links(week: paths.Week) -> int:
    """Write the week's `links.parquet` from its `signals.parquet`, and return how many requests have a parent."""
    links = link_requests(read_requests(week.folder / paths.SIGNALS_FILE))
    parallel.write_rows(links, LINKS_SCHEMA, week.folder / paths.LINKS_FILE)
    return sum(1 for link in links if link["parent_request_id"] is not None)
