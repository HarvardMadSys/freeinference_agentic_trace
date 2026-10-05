"""The sessions the release leaves out, from `export/configs/exclusions.toml`.

A session is excluded by its harness or by its id as the release gives it.
Every exclusion has a reason, which the week's manifest counts.
"""

import dataclasses
import pathlib
import tomllib

from export.privacy import pseudonyms


class ExclusionsError(Exception):
    """The exclusions file is malformed."""


@dataclasses.dataclass(frozen=True)
class Exclusions:
    """Excluded harnesses and release session ids, each with its reason."""

    harnesses: dict[str, str]
    sessions: dict[str, str]


def load_exclusions(path: pathlib.Path) -> Exclusions:
    """The exclusions of the file; every entry must name what it excludes and why."""
    with open(path, "rb") as handle:
        table = tomllib.load(handle)
    harnesses = {}
    for entry in table.get("harness", []):
        require_keys(entry, ("name", "reason"), path)
        harnesses[entry["name"]] = entry["reason"]
    sessions = {}
    for entry in table.get("session", []):
        require_keys(entry, ("id", "reason"), path)
        sessions[entry["id"]] = entry["reason"]
    return Exclusions(harnesses=harnesses, sessions=sessions)


def require_keys(entry: dict, keys: tuple[str, ...], path: pathlib.Path) -> None:
    """Stop unless the entry has every key."""
    missing = [key for key in keys if key not in entry]
    if missing:
        raise ExclusionsError(f"{path}: an entry lacks {missing}: {entry}")


def reason_excluded(session: dict, exclusions: Exclusions, salt: bytes) -> str | None:
    """Why one of the week's sessions is left out of the release, or None when it is kept."""
    if session["harness"] in exclusions.harnesses:
        return exclusions.harnesses[session["harness"]]
    return exclusions.sessions.get(pseudonyms.session_hash(session["session_id"], salt))
