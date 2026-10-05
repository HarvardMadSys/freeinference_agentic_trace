"""Steps, turns, durations and waits of sessions.

A user step is a request whose `step_trigger` is `user`, a tool step one whose
`step_trigger` is `tool`; a session's first request is always a user step. A
turn is a user step and the tool steps after it. Requests must be in session
and step order.
"""

import pandas

from release import schema


def counts_per_session(requests: pandas.DataFrame) -> pandas.DataFrame:
    """Per session: its requests, user steps and tool steps."""
    steps = requests.assign(is_user_step=requests.step_trigger == "user", is_tool_step=requests.step_trigger == "tool")
    return steps.groupby("session_id").agg(
        requests=("step", "size"),
        user_steps=("is_user_step", "sum"),
        tool_steps=("is_tool_step", "sum"),
    )


def turn_numbers(requests: pandas.DataFrame) -> pandas.Series:
    """Each request's turn within its session, counted from 1 at each user step."""
    is_user_step = (requests.step_trigger == "user").astype(int)
    return is_user_step.groupby(requests.session_id).cumsum()


def turn_durations(requests: pandas.DataFrame) -> pandas.Series:
    """Each turn's duration in seconds: its last completion minus its first arrival."""
    turn_steps = requests.assign(turn=turn_numbers(requests)).groupby(["session_id", "turn"])
    return (turn_steps.end_ms.max() - turn_steps.start_ms.min()) / schema.MILLISECONDS_PER_SECOND


def session_durations(requests: pandas.DataFrame) -> pandas.Series:
    """Each session's duration in seconds: its last completion minus its first arrival."""
    sessions = requests.groupby("session_id")
    return (sessions.end_ms.max() - sessions.start_ms.min()) / schema.MILLISECONDS_PER_SECOND


def waits(requests: pandas.DataFrame) -> pandas.Series:
    """Each request's wait in seconds: its arrival minus the previous request's completion; none at step 0."""
    previous_end_ms = requests.groupby("session_id").end_ms.shift(1)
    return (requests.start_ms - previous_end_ms) / schema.MILLISECONDS_PER_SECOND
