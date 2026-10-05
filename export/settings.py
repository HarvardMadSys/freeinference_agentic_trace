"""The maintainer's settings for this export, read from `private/export.toml`: its weeks, its process limit, its excluded accounts.

`settings.example.toml`, beside this module, is the template, with no real account.
"""

import dataclasses
import datetime
import pathlib
import tomllib

SUNDAY = 6  # `date.weekday()` of a Sunday; every week starts on one


class SettingsError(Exception):
    """The settings file is missing a setting or holds a wrong one."""


@dataclasses.dataclass(frozen=True)
class ExportSettings:
    """The weeks of an export, the most processes one step may use, and whose requests are dropped."""

    weeks: tuple[datetime.date, ...]
    max_processes: int
    excluded_users: frozenset[str]


def load_export_settings(path: pathlib.Path) -> ExportSettings:
    """Read and check a settings file."""
    with open(path, "rb") as handle:
        settings = tomllib.load(handle)
    return ExportSettings(
        weeks=read_weeks(required_setting(settings, "weeks", path)),
        max_processes=read_max_processes(required_setting(settings, "max_processes", path)),
        excluded_users=frozenset(required_setting(settings, "excluded_users", path)),
    )


def required_setting(settings: dict, name: str, path: pathlib.Path):
    """The value of a setting that every settings file must have."""
    if name not in settings:
        raise SettingsError(f"{path} has no `{name}` setting")
    return settings[name]


def read_weeks(values: list) -> tuple[datetime.date, ...]:
    """The weeks, each given by its Sunday as a TOML date."""
    weeks = []
    for value in values:
        if type(value) is not datetime.date:
            raise SettingsError(f"week {value!r} is not a date such as 2026-05-17")
        if value.weekday() != SUNDAY:
            raise SettingsError(f"week {value} does not start on a Sunday")
        weeks.append(value)
    return tuple(weeks)


def read_max_processes(value: object) -> int:
    """The most processes one step may use, at least 1."""
    if type(value) is not int or value < 1:
        raise SettingsError(f"max_processes must be a whole number of at least 1, not {value!r}")
    return value
