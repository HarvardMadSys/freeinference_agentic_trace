"""Tool use along a turn: which kind of work the calls do early in a turn, and which late.

A turn is a user step and the tool steps after it. Its requests that made a
call are numbered k = 0 … n−1 in step order, and request k falls in bin
min(⌊10 (k + 0.5) / n⌋, 9); each of its calls falls in that bin. Calls are
put in eight groups by their category; a `*File Ops` call is Write/Edit when
any program it ran changes files, else Read when one only looks.
"""

import numpy
import pandas

from paper.metrics import turns

BINS = 10  # the trajectory is cut into tenths
GROUPS = ("Read", "Search", "Write/Edit", "Execute", "Build/Test", "Git/VCS", "Browser/Web", "Other")
GROUP_OF_CATEGORY = {
    "Read": "Read", "*Search": "Search", "Edit": "Write/Edit", "Write": "Write/Edit", "*Interpreter": "Execute",
    "*Build/Test": "Build/Test", "*Git/VCS": "Git/VCS", "Browser": "Browser/Web", "*Network": "Browser/Web",
}
FILE_OPS = "*File Ops"
# The `*File Ops` programs that only look, and those that change files.
READING_PROGRAMS = frozenset({
    "ls", "dir", "cat", "head", "tail", "wc", "du", "df", "stat", "file", "tree", "pwd", "date", "ps", "top", "htop",
    "env", "printenv", "whoami", "uname", "which", "whereis", "less", "more", "realpath", "basename", "dirname",
    "readlink", "strings", "hexdump", "xxd", "od", "nl", "column", "diff", "cmp", "md5sum", "sha256sum", "cksum", "id",
    "groups", "hostname", "uptime", "free", "history", "cd", "type", "lsof", "tasklist", "pgrep", "nproc", "getconf",
    "get-content", "get-childitem", "test-path", "select-object", "where-object", "sort-object", "measure-object",
    "format-table", "format-list", "out-string", "write-output", "write-host", "resolve-path", "convertfrom-json",
    "convertto-json", "get-command", "get-location", "get-process", "get-item", "get-date", "get-help", "get-member",
    "compare-object", "foreach-object", "get-service", "get-itemproperty", "sort", "uniq",
})
WRITING_PROGRAMS = frozenset({
    "mkdir", "rm", "rmdir", "mv", "cp", "touch", "chmod", "chown", "chgrp", "ln", "tee", "truncate", "dd", "shred",
    "tar", "zip", "unzip", "gzip", "gunzip", "7z", "patch", "mktemp", "install", "rename", "split", "new-item",
    "remove-item", "set-content", "add-content", "copy-item", "move-item", "rename-item", "out-file", "clear-content",
    "set-itemproperty", "new-directory", "del", "ren", "md", "rd", "ni", "mklink",
})


def first_words(programs) -> set[str]:
    """The first word of each program label a call ran, lower-cased."""
    words = set()
    for label in programs if programs is not None else []:
        if label.split():
            words.add(label.split()[0].lower())
    return words


def group_of_call(category: str, programs) -> str:
    """The trajectory group of one call; a `*File Ops` call's by its programs, a writing one first."""
    if category != FILE_OPS:
        return GROUP_OF_CATEGORY.get(category, "Other")
    words = first_words(programs)
    if words & WRITING_PROGRAMS:
        return "Write/Edit"
    if words & READING_PROGRAMS:
        return "Read"
    return "Other"


def trajectory_bins(requests: pandas.DataFrame) -> pandas.Series:
    """Each request's trajectory bin within its turn; missing for a request that made no call.

    Requests must be in session and step order.
    """
    calling = requests[requests.n_tool_calls > 0]
    turn = turns.turn_numbers(requests)[calling.index]
    keys = [calling.session_id, turn]
    position = calling.groupby(keys).cumcount()
    count = calling.groupby(keys).step.transform("size")
    bins = numpy.minimum(numpy.floor(BINS * (position + 0.5) / count), BINS - 1).astype(int)
    return pandas.Series(bins, index=calling.index).reindex(requests.index)


def trajectory_shares(tool_calls: pandas.DataFrame, requests: pandas.DataFrame) -> pandas.DataFrame:
    """Per bin (rows, 0–9) and group (columns), the share of the bin's calls; each row sums to 1."""
    binned = requests.assign(trajectory_bin=trajectory_bins(requests))[["session_id", "step", "trajectory_bin"]]
    calls = tool_calls.merge(binned.dropna(), on=["session_id", "step"], how="inner")
    groups = [group_of_call(category, programs) for category, programs in zip(calls.category, calls.programs)]
    counts = pandas.crosstab(calls.trajectory_bin.astype(int), pandas.Series(groups, index=calls.index))
    counts = counts.reindex(index=range(BINS), columns=list(GROUPS), fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0).fillna(0.0)

