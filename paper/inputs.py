"""What the figures are drawn from: the loaded release, where it lives, and the simulation's tallies."""

import dataclasses
import pathlib

from release import reader


@dataclasses.dataclass(frozen=True)
class Inputs:
    """The release's requests and sessions, the release folder (for its other files), and the simulation folder."""

    release: reader.Release
    release_directory: pathlib.Path
    simulation_directory: pathlib.Path
