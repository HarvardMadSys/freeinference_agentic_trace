"""The paper's tables and figures: `python -m paper [--only ITEM]`.

Every released week is read from `data/v1/release` and the simulation from
`data/v1/simulation`; the figures go to `figures/v1`, one folder per section
of the paper (`dataset/`, `sessions/`, `tools/`, `prefix_caching/`,
`benchmark_vs_production/`), with `numbers.json`, every number the paper's
text quotes, and `README.md`, the index from each paper item to its files and
its function. `--only` draws the section of one item by its paper name.
"""

import argparse
import json
import pathlib

import paper
from paper import benchmark_vs_production, dataset, inputs, prefix_caching, sessions, tools
from release import reader

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = REPOSITORY_ROOT / "data" / "v1" / "release"
DEFAULT_SIMULATION = REPOSITORY_ROOT / "data" / "v1" / "simulation"  # tracked: the simulation ships with the repository
DEFAULT_OUTPUT = REPOSITORY_ROOT / "figures" / "v1"

# The paper's sections in order, each drawn into its own folder of the output.
SECTIONS = {
    "dataset": dataset,
    "sessions": sessions,
    "tools": tools,
    "prefix_caching": prefix_caching,
    "benchmark_vs_production": benchmark_vs_production,
}


def draw_figures(figure_inputs: inputs.Inputs, output_directory: pathlib.Path, sections: dict = SECTIONS) -> dict:
    """Every table and figure the inputs allow, one folder per section, and their numbers by section."""
    return {name: section.draw(figure_inputs, output_directory / name) for name, section in sections.items()}


def sections_of(paper_item: str) -> dict:
    """The section a paper item is drawn in, by its name in the registry; stops, listing the items, for an unknown one."""
    items = paper.items_of(paper_item)
    if not items:
        known = ", ".join(dict.fromkeys(item.paper for item in paper.FIGURES))
        raise SystemExit(f"{paper_item!r} is not in the figure registry; the items are {known}")
    return {item.section: SECTIONS[item.section] for item in items}


def parse_arguments(arguments: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m paper", description="The paper's tables and figures, from the release.")
    parser.add_argument("--only", metavar="ITEM", help="one paper item by its name in the registry (figures/v1/README.md); its section is drawn")
    parser.add_argument("--release", type=pathlib.Path, default=DEFAULT_RELEASE, metavar="DIR", help="the release directory")
    parser.add_argument("--simulation", type=pathlib.Path, default=DEFAULT_SIMULATION, metavar="DIR", help="the simulation directory")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT, metavar="DIR", help="where the figures go")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> None:
    """Draw every section, or one item's section; write `numbers.json` and the index when every section was drawn."""
    options = parse_arguments(arguments)
    figure_inputs = inputs.Inputs(release=reader.load(options.release), release_directory=options.release,
                                  simulation_directory=options.simulation)
    if options.only is not None:
        draw_figures(figure_inputs, options.output, sections_of(options.only))
        files = ", ".join(f"{item.section}/{stem}" for item in paper.items_of(options.only) for stem in item.outputs)
        print(f"drew {options.only}: {files or 'numbers only'} into {options.output}")
        return
    numbers = draw_figures(figure_inputs, options.output)
    with open(options.output / "numbers.json", "w") as handle:
        json.dump(numbers, handle, indent=2)
        handle.write("\n")
    (options.output / "README.md").write_text(paper.index_text())
    print(f"drew {', '.join(numbers)} into {options.output}, with numbers.json and README.md")


if __name__ == "__main__":
    main()
