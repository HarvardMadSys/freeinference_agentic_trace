"""The release's pseudonyms: keyed hashes of the raw ids, with the one secret salt, and each provider's letter.

The same raw id hashes to the same value in every release, so later releases
join onto earlier ones. A session's id and a trace's id use the same rule, so
a trace's id equals its top-level session's id. A provider is named only by
its letter, from a private map kept for every release; a provider the map
does not list stops the release, so that a new one is added on purpose.
"""

import dataclasses
import hashlib
import pathlib
import tomllib

from export import paths

SESSION_DIGEST_BYTES = 16  # a session or trace id: 32 hexadecimal characters
USER_DIGEST_BYTES = 8  # an account id: 16 hexadecimal characters
SERIES_ID_CHARACTERS = 16  # how much of the salt's sha256 names its series


@dataclasses.dataclass(frozen=True)
class PrivateKeys:
    """What the release step must have and no one else may: the salt, and each provider's letter."""

    salt: bytes
    provider_letters: dict[str, str]


def load_private_keys(salt_path: pathlib.Path, providers_path: pathlib.Path) -> PrivateKeys:
    """The maintainer's private keys, from `private/`."""
    return PrivateKeys(salt=load_salt(salt_path), provider_letters=load_letters(providers_path))


class MissingSalt(Exception):
    """The repository has no `private/salt`."""


def load_salt(path: pathlib.Path) -> bytes:
    """The salt, stored as hexadecimal; an error when it is missing, never a new one."""
    if not path.exists():
        raise MissingSalt(
            f"{path} does not exist: restore the salt from its backup. "
            "A new salt would give every id of every release a new value."
        )
    return bytes.fromhex(path.read_text().strip())


def salt_series(salt: bytes) -> str:
    """The salt's public name: the start of its sha256, which reveals nothing of the salt."""
    return hashlib.sha256(salt).hexdigest()[:SERIES_ID_CHARACTERS]


def session_hash(raw_session_id: str, salt: bytes) -> str:
    """The hashed id of a session, or of the trace it heads."""
    digest = hashlib.blake2b(b"session:" + raw_session_id.encode(), digest_size=SESSION_DIGEST_BYTES, key=salt)
    return digest.hexdigest()


def user_hash(raw_user_id: str | None, salt: bytes) -> str | None:
    """The hashed id of an account; None when the account is unknown."""
    if raw_user_id is None:
        return None
    digest = hashlib.blake2b(b"user:" + raw_user_id.encode(), digest_size=USER_DIGEST_BYTES, key=salt)
    return digest.hexdigest()


class UnlistedProvider(Exception):
    """A request's provider has no letter in the private map."""


def load_letters(path: pathlib.Path) -> dict[str, str]:
    """The letter of each provider."""
    with open(path, "rb") as handle:
        return dict(tomllib.load(handle)["letters"])


def letter_of(provider: str | None, letters: dict[str, str]) -> str | None:
    """The provider's letter; None for a request that names no provider."""
    if provider is None:
        return None
    if provider not in letters:
        raise UnlistedProvider(f"provider {provider!r} has no letter; add it to private/providers.toml with the next free "
                               f"letter (the format is {paths.PROVIDERS_EXAMPLE.name}, beside the settings example)")
    return letters[provider]
