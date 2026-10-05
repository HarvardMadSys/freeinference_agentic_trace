"""A table set as text, the way the paper sets its tables: rules, a title, and rows of cells.

A table is a list of lines, top to bottom, each a kind and its cells:
`rule` (a horizontal line, no cells), `title` (one centred cell, bold), and
any other kind, whose cells sit at that kind's column positions; a kind named
in `bold_kinds` is set in bold.
"""

import pathlib

from matplotlib import lines

from paper.plot_style import style

TABLE_FONT = 9.0  # points: the paper's table font size
TEXT_ROW_HEIGHT = 0.19  # inches
RULE_ROW_HEIGHT = 0.07  # inches


def draw_table(rows: list[tuple[str, list[str]]], columns: dict[str, tuple], width: float, bold_kinds: frozenset[str],
               output_directory: pathlib.Path, name: str) -> None:
    """Set the lines as text and rules, top to bottom, on a page `width` inches wide, and save it as `name`.

    `columns` gives, per kind of line, each cell's position (a share of the width) and alignment.
    """
    heights = [RULE_ROW_HEIGHT if kind == "rule" else TEXT_ROW_HEIGHT for kind, _ in rows]
    figure = style.new_page((width, sum(heights)))
    top = sum(heights)
    for (kind, cells), height in zip(rows, heights):
        middle = (top - height / 2) / sum(heights)
        top -= height
        if kind == "rule":
            figure.add_artist(lines.Line2D([0.01, 0.99], [middle, middle], color=style.INK, lw=0.6))
            continue
        if kind == "title":
            figure.text(0.5, middle, cells[0], ha="center", va="center", fontsize=TABLE_FONT, fontweight="bold")
            continue
        weight = "bold" if kind in bold_kinds else "normal"
        for text, (position, align) in zip(cells, columns[kind]):
            figure.text(position, middle, text, ha=align, va="center", fontsize=TABLE_FONT, fontweight=weight)
    style.save(figure, output_directory, name)
