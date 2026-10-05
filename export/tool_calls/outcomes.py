"""Whether one tool call succeeded, decided from the start of its result.

There is no standard way to report a tool failure, so the strongest evidence
decides, in this order:
1. the harness flagged the result as an error;
2. the result is JSON with a status, an exit code, a success flag or an
   error field;
3. a `<returncode>` wrapper at the start;
4. a denial, timeout or failure marker near the start;
5. a literal `Exit code: N` near the start;
6. the harness flagged the result as not an error;
7. the first line starts with the word "error";
8. otherwise the call succeeded.
The harness's flag of "not an error" is below the text tests on purpose: a
harness can set it while the command it ran still failed.
"""

import dataclasses
import pathlib
import re
import tomllib

import orjson

OK = "ok"
ERROR = "error"
DENIED = "denied"
TIMEOUT = "timeout"
OUTCOMES = (OK, ERROR, DENIED, TIMEOUT)

RETURN_CODE = re.compile(r'^\s*(?:<returncode>(-?\d+)</returncode>|\{[^{}]{0,200}"returncode"\s*:\s*(-?\d+))')
EXIT_CODE_IN_TEXT = re.compile(r"[Ee]xit [Cc]ode:?\s*(-?\d+)")
ERROR_FIRST_LINE = re.compile(r"^\s*(?:error|ERROR|Error)\b[:\s]")


@dataclasses.dataclass(frozen=True)
class OutcomeRules:
    """The markers and lengths of `export/configs/outcome_rules.toml`."""

    head_characters: int
    marker_characters: int
    denial_markers: tuple[str, ...]
    timeout_markers: tuple[str, ...]
    failure_markers: tuple[str, ...]
    empty_error_values: frozenset[str]
    success_statuses: frozenset[str]


def load_rules(path: pathlib.Path) -> OutcomeRules:
    """The outcome rules."""
    with open(path, "rb") as handle:
        table = tomllib.load(handle)
    return OutcomeRules(
        head_characters=table["head_characters"],
        marker_characters=table["marker_characters"],
        denial_markers=tuple(table["denial_markers"]),
        timeout_markers=tuple(table["timeout_markers"]),
        failure_markers=tuple(table["failure_markers"]),
        empty_error_values=frozenset(table["empty_error_values"]),
        success_statuses=frozenset(table["success_statuses"]),
    )


def json_outcome(head: str, outcome_rules: OutcomeRules) -> str | None:
    """The outcome a JSON result states in its status, exit code, success flag or error field; None if none."""
    if not head.startswith("{"):
        return None
    try:
        body = orjson.loads(head)
    except orjson.JSONDecodeError:
        return None
    if not isinstance(body, dict):
        return None
    status = str(body.get("status") or "").lower()
    if status == "denied":
        return DENIED
    if status == "error":
        return ERROR
    exit_code = body.get("exit_code", body.get("returncode"))
    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
        return OK if exit_code == 0 else ERROR
    if body.get("success") is False:
        return ERROR
    if body.get("success") is True:
        return OK
    error = body.get("error")
    if isinstance(error, str) and error.strip().lower() in outcome_rules.empty_error_values:
        error = None
    if error is not None:
        return ERROR
    return OK if status in outcome_rules.success_statuses else None


def marker_outcome(head: str, outcome_rules: OutcomeRules) -> str | None:
    """The outcome a harness's wrapper phrase states near the start of the result; None if none."""
    opening = head[:outcome_rules.marker_characters]
    if any(marker in opening for marker in outcome_rules.denial_markers):
        return DENIED
    if any(marker in opening for marker in outcome_rules.timeout_markers):
        return TIMEOUT
    if any(marker in opening for marker in outcome_rules.failure_markers):
        return ERROR
    return None


def outcome(text: str, is_error: object, outcome_rules: OutcomeRules) -> str:
    """The outcome of one tool result, from its text and the harness's error flag, if any."""
    head = text[:outcome_rules.head_characters].lstrip()
    if is_error is True:
        return ERROR
    stated = json_outcome(head, outcome_rules)
    if stated is not None:
        return stated
    return_code = RETURN_CODE.match(head)
    if return_code:
        return OK if int(return_code.group(1) or return_code.group(2)) == 0 else ERROR
    stated = marker_outcome(head, outcome_rules)
    if stated is not None:
        return stated
    exit_code = EXIT_CODE_IN_TEXT.search(head[:outcome_rules.marker_characters])
    if exit_code:
        return OK if int(exit_code.group(1)) == 0 else ERROR
    if is_error is False:
        return OK
    if ERROR_FIRST_LINE.match(head.split("\n", 1)[0]):
        return ERROR
    return OK
