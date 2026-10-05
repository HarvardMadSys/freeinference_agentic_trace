"""The keys of a tool-call id: the forms under which one id is recognised again.

A model emits an id; a harness may replay it renumbered (`_<digits>` appended)
or with a `toolu` shim prefixed. A key is an id without punctuation. The
emitted keys of an id are the keys it may be replayed under; the replayed keys
of a replayed id are the keys it may have been emitted under; the own keys
of an id never allow renumbering. Two ids match when their key sets meet.
"""
import re


NON_ALPHANUMERIC = re.compile(r"[^A-Za-z0-9]")
# A final `_<digits>` that a harness renumbers, dropped only when 12 or more characters come before it.
SEQUENCE_SUFFIX = re.compile(r"(.{12,}?)_\d+")
REPLAY_SHIM_PREFIX = "toolu"  # a prefix some harnesses add to the tool-call ids they replay


def own_keys(raw_id: object) -> set[str]:
    """The keys of an id as written, never renumbered: without punctuation, and that without a leading `toolu`."""
    if not isinstance(raw_id, str):
        return set()
    key = NON_ALPHANUMERIC.sub("", raw_id)
    keys = {key}
    if key.startswith(REPLAY_SHIM_PREFIX):
        keys.add(key[len(REPLAY_SHIM_PREFIX):])
    keys.discard("")
    return keys


def emitted_keys(raw_id: object) -> set[str]:
    """The keys of an id the model emitted: without punctuation, and without a `_<digits>` suffix."""
    if not isinstance(raw_id, str) or not raw_id:
        return set()
    keys = {NON_ALPHANUMERIC.sub("", raw_id)}
    suffixed = SEQUENCE_SUFFIX.fullmatch(raw_id)
    if suffixed is not None:
        keys.add(NON_ALPHANUMERIC.sub("", suffixed.group(1)))
    keys.discard("")
    return keys


def replayed_keys(raw_id: object) -> set[str]:
    """The keys of an id a harness replayed: the emitted keys, and each without a leading `toolu`."""
    keys = emitted_keys(raw_id)
    for key in list(keys):
        if key.startswith(REPLAY_SHIM_PREFIX):
            keys.add(key[len(REPLAY_SHIM_PREFIX):])
    keys.discard("")
    return keys
