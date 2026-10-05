"""Reading one log record: the line, the fields later steps use, and when the request completed.

A value the record does not carry, or carries as null or with the wrong JSON
type, is read as None, never as 0, false or an empty string. A completion
time is an integer count of milliseconds since the Unix epoch.
"""
import datetime
import math
import re

import orjson


# `YYYY-MM-DD` then `T` or a space, `hh:mm:ss`, an optional fraction, and an
# optional `Z` or `+HH:MM` / `-HH:MM` offset. ASCII digits only.
TIMESTAMP_PATTERN = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})[T ]([0-9]{2}):([0-9]{2}):([0-9]{2})"
    r"(?:\.([0-9]+))?"
    r"(Z|[+-][0-9]{2}:[0-9]{2})?"
)

UNIX_EPOCH = datetime.date(1970, 1, 1)
SECONDS_PER_DAY = 86_400
MILLISECONDS_PER_SECOND = 1_000
MILLISECONDS_PER_DAY = SECONDS_PER_DAY * MILLISECONDS_PER_SECOND
MILLISECOND_DIGITS = 3  # fraction digits kept; later digits are truncated, not rounded
DAYS_PER_WEEK = 7


def completion_ms(timestamp: object) -> int | None:
    """A record's `timestamp` in milliseconds since the epoch, truncated.

    None when the value is not a string of the timestamp form, or names a
    date or time that does not exist. A timestamp without an offset is UTC.
    """
    if not isinstance(timestamp, str):
        return None
    match = TIMESTAMP_PATTERN.fullmatch(timestamp)
    if match is None:
        return None
    year, month, day, hour, minute, second, fraction, offset = match.groups()
    try:
        date = datetime.date(int(year), int(month), int(day))
    except ValueError:
        return None
    if int(hour) > 23 or int(minute) > 59 or int(second) > 59:
        return None
    offset_seconds = offset_in_seconds(offset)
    if offset_seconds is None:
        return None
    seconds = (
        (date - UNIX_EPOCH).days * SECONDS_PER_DAY
        + int(hour) * 3_600
        + int(minute) * 60
        + int(second)
        - offset_seconds
    )
    milliseconds = int((fraction or "").ljust(MILLISECOND_DIGITS, "0")[:MILLISECOND_DIGITS])
    return seconds * MILLISECONDS_PER_SECOND + milliseconds


def offset_in_seconds(offset: str | None) -> int | None:
    """Seconds east of UTC for `Z`, `+HH:MM` or `-HH:MM`; 0 when absent; None when invalid."""
    if offset is None or offset == "Z":
        return 0
    hours = int(offset[1:3])
    minutes = int(offset[4:6])
    if hours > 23 or minutes > 59:
        return None
    seconds = hours * 3_600 + minutes * 60
    if offset.startswith("-"):
        return -seconds
    return seconds


def week_of(end_ms: int) -> datetime.date:
    """The Sunday that starts the UTC week, Sunday 00:00 to the next Sunday 00:00, holding `end_ms`."""
    day = UNIX_EPOCH + datetime.timedelta(days=end_ms // MILLISECONDS_PER_DAY)
    days_since_sunday = (day.weekday() + 1) % DAYS_PER_WEEK
    return day - datetime.timedelta(days=days_since_sunday)


# Where each token count is looked up, in order; the first integer found wins.
# `record` is the raw record, `usage` the response's `usage` object, and
# `details` its `prompt_tokens_details` object.
TOKEN_COUNT_SOURCES = {
    "prompt_tokens": [("record", "prompt_tokens"), ("usage", "prompt_tokens"), ("usage", "input_tokens")],
    "completion_tokens": [("record", "completion_tokens"), ("usage", "completion_tokens"), ("usage", "output_tokens")],
    "reasoning_tokens": [("record", "reasoning_tokens"), ("usage", "reasoning_tokens")],
    "cache_read_tokens": [("record", "cache_read_tokens"), ("details", "cached_tokens"), ("usage", "cache_read_input_tokens")],
    "cache_write_tokens": [("record", "cache_write_tokens"), ("usage", "cache_creation_input_tokens")],
}


def parse_line(line: bytes) -> dict | None:
    """The record a raw line holds; None when the line is not a JSON object."""
    return parse_object(line)


def parse_object(text: bytes | str) -> dict | None:
    """The JSON object in `text`; None when it is not JSON or not an object."""
    try:
        value = orjson.loads(text)
    except orjson.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    return value


def read_string(value: object) -> str | None:
    """The value when it is a string."""
    if isinstance(value, str):
        return value
    return None


def read_integer(value: object) -> int | None:
    """The value when it is a finite number, not a boolean; a fraction is truncated toward zero."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return None


def read_boolean(value: object) -> bool | None:
    """The value when it is a boolean."""
    if isinstance(value, bool):
        return value
    return None


def read_body(value: object) -> str | None:
    """A body (`prompt`, `response`, `tools`) exactly as recorded, when it is a non-empty string."""
    if isinstance(value, str) and value:
        return value
    return None


def user_agent(record: dict) -> str | None:
    """The `user_agent` inside the record's `metadata` JSON text; no other metadata key is read."""
    metadata = read_body(record.get("metadata"))
    if metadata is None:
        return None
    parsed = parse_object(metadata)
    if parsed is None:
        return None
    return read_string(parsed.get("user_agent"))


def token_counts(record: dict) -> dict[str, int | None]:
    """The five token counts: the record's own field first, then the response's usage."""
    places = {"record": record, "usage": {}, "details": {}}
    response = read_body(record.get("response"))
    parsed_response = parse_object(response) if response is not None else None
    if parsed_response is not None and isinstance(parsed_response.get("usage"), dict):
        places["usage"] = parsed_response["usage"]
        if isinstance(places["usage"].get("prompt_tokens_details"), dict):
            places["details"] = places["usage"]["prompt_tokens_details"]
    counts = {}
    for field, sources in TOKEN_COUNT_SOURCES.items():
        counts[field] = None
        for place, key in sources:
            count = read_integer(places[place].get(key))
            if count is not None:
                counts[field] = count
                break
    return counts


def kept_fields(record: dict, end_ms: int) -> dict:
    """The fields of a record that later stages use, with times in integer milliseconds."""
    latency_ms = read_integer(record.get("latency_ms"))
    fields = {
        "request_id": read_string(record.get("request_id")),
        "user_id": read_string(record.get("user_id")),
        "end_ms": end_ms,
        "start_ms": end_ms - (latency_ms or 0),
        "latency_ms": latency_ms,
        "ttft_ms": read_integer(record.get("ttft_ms")),
        "stream": read_boolean(record.get("stream")),
        "model_id": read_string(record.get("model_id")),
        "provider": read_string(record.get("provider")),
    }
    fields.update(token_counts(record))
    fields["prompt"] = read_body(record.get("prompt"))
    fields["response"] = read_body(record.get("response"))
    fields["tools"] = read_body(record.get("tools"))
    return fields
