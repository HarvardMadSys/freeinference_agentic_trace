"""The release on disk: `python -m release download --tables | --traces [--week SUNDAY ...]` and `convert [--week SUNDAY ...]`.

`download` fetches the release from Hugging Face and lays it out as
`data/v1/release`: the tables, the trace files decompressed, or both.
`convert` turns released weeks' trace lines into their three tables: the
listed weeks, or every week whose trace files are downloaded.
"""

import argparse
import pathlib

from release import convert, download, reader

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = REPOSITORY_ROOT / "data" / "v1" / "release"
DEFAULT_HUB = REPOSITORY_ROOT / "data" / "hub"  # where a download keeps the Hub's files before laying them out
RELEASE_OPTION = {"type": pathlib.Path, "default": DEFAULT_RELEASE, "metavar": "DIR", "help": "the release directory"}


def run_download(options: argparse.Namespace) -> None:
    """Fetch the release, or some weeks of it, from Hugging Face and lay it out as `data/v1/release`."""
    if not options.tables and not options.traces:
        raise SystemExit("say what to download: --tables (9 GB), --traces (108 GB), or both")
    try:
        weeks = download.download(options.repo, options.hub, options.release, options.tables, options.traces, options.week)
    except download.DownloadError as error:
        raise SystemExit(str(error))
    what = " and ".join(name for name, chosen in (("tables", options.tables), ("traces", options.traces)) if chosen)
    print(f"{len(weeks)} weeks' {what} laid out under {options.release}, from {options.repo}")


def run_convert(options: argparse.Namespace) -> None:
    """Convert the listed weeks, or every week whose trace files are downloaded, into their three tables."""
    try:
        weeks = convert.weeks_to_convert(options.release.expanduser(), options.week)
    except (convert.ConvertError, reader.ReleaseError) as error:
        raise SystemExit(str(error))
    for week_directory in weeks.without_traces:
        print(f"{week_directory.name}: left out, its trace files are not downloaded")
    for week_directory in weeks.with_traces:
        record = convert.convert_week(week_directory)
        rows = ", ".join(f"{facts['rows']} rows in {name}" for name, facts in record["files"].items())
        print(f"{week_directory.name}: {rows}")


COMMANDS = {"download": run_download, "convert": run_convert}


def parse_arguments(arguments: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m release", description="The anonymous release on disk.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    download_command = commands.add_parser("download", help="fetch the release from Hugging Face into the layout the "
                                           "code reads: the tables, the trace files, or both")
    download_command.add_argument("--tables", action="store_true", help="the tables and tables.json (9 GB)")
    download_command.add_argument("--traces", action="store_true", help="the trace files, decompressed with zstd (108 GB)")
    download_command.add_argument("--week", action="append", metavar="SUNDAY",
                                  help="one week's Sunday; repeat for several; every week without it")
    download_command.add_argument("--repo", default=download.REPOSITORY, help="the Hub repository")
    download_command.add_argument("--hub", type=pathlib.Path, default=DEFAULT_HUB, metavar="DIR", help="where the Hub's files are kept")
    download_command.add_argument("--release", **RELEASE_OPTION)
    convert_command = commands.add_parser("convert", help="convert released weeks' trace lines into their three tables")
    convert_command.add_argument("--week", action="append", metavar="SUNDAY",
                                 help="one week's Sunday, YYYY-MM-DD; repeat for several; without it, every week whose "
                                 "trace files are downloaded")
    convert_command.add_argument("--release", **RELEASE_OPTION)
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> None:
    options = parse_arguments(arguments)
    COMMANDS[options.command](options)


if __name__ == "__main__":
    main()
