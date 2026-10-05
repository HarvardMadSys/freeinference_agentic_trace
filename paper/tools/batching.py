"""Tool batching: the batching profile, latency by batch width, and the calls per request of each category."""

import pathlib

import pandas

from paper.plot_style import axes, marks, style, tables
from paper.metrics import tools

PROFILE_POSITIONS = ((0.04, "left"), (0.42, "left"), (0.96, "right"))
TABLE_WIDTH = 2.4  # inches: a half column, as in the paper


def batching_profile(calls: pandas.DataFrame, session_count: int, output_directory: pathlib.Path) -> dict:
    """The batching profile: sessions with a batch, the calls per request, and the batch mix, in percent."""
    profile = tools.batching_profile(calls, session_count)
    rows = [("Sessions", "Batched", profile["sessions_batched"]), ("", "Unbatched", 100 - profile["sessions_batched"])]
    for position, bucket in enumerate(tools.WIDTH_BUCKETS):
        metric = "Tool calls" if position == 0 else ""
        rows.append((metric, f"{bucket} call" + ("" if bucket == "1" else "s"), profile["calls_per_request"][bucket]))
    rows += [("Batch mix", "Same tool", profile["same_tool"]), ("", "Mixed", profile["mixed"])]
    lines = [("rule", []), ("header", ["Metric", "Category", "Share (%)"]), ("rule", [])]
    lines += [("row", [metric, label, f"{share:.1f}"]) for metric, label, share in rows]
    lines.append(("rule", []))
    tables.draw_table(lines, {"header": PROFILE_POSITIONS, "row": PROFILE_POSITIONS}, TABLE_WIDTH, frozenset({"header"}),
                      output_directory, "batching_profile")
    pandas.DataFrame(rows, columns=["metric", "category", "share_percent"]).to_csv(output_directory / "batching_profile.csv",
                                                                                 index=False)
    shares = {"sessions_batched": profile["sessions_batched"], **profile["calls_per_request"],
              "same_tool": profile["same_tool"], "mixed": profile["mixed"]}
    return {f"{name}_percent": float(value) for name, value in shares.items()}


def latency_by_width(calls: pandas.DataFrame, output_directory: pathlib.Path) -> dict:
    """The tool wait of a batch by its width, against issuing the calls one per request."""
    by_width = tools.latency_by_width(calls)
    figure, plot = style.new_figure()
    color = style.BATCH_COLOR["Batched Tool Latency"]
    band = style.GREEN
    plot.plot(by_width.width, by_width.serial_s, color=style.BATCH_COLOR["Serial + LLM Turn"], lw=2.4, ls="--",
              label="Serial + LLM Turn", zorder=4)
    plot.fill_between(by_width.width, by_width.p10, by_width.p90, color=band, alpha=0.12, lw=0)
    plot.fill_between(by_width.width, by_width.p25, by_width.p75, color=band, alpha=0.3, lw=0)
    plot.plot(by_width.width, by_width.p50, color=color, lw=2.4, label="Batched Tool Latency", zorder=4)
    names = [str(width) for width in by_width.width[:-1]] + [f"{by_width.width.iloc[-1]}+"]
    axes.category_axis(plot, "x", list(by_width.width), names)
    plot.set_xlim(by_width.width.min() - 0.3, by_width.width.max() + 0.3)
    plot.set_xlabel("Calls")
    high = max(float(by_width.serial_s.max()), float(by_width.p90.max())) * 1.05
    axes.apply(plot, "y", "seconds", "latency", low=0, high=high)
    marks.legend_inside(plot, "upper left")
    style.save(figure, output_directory, "latency_by_batch_width")
    p90 = by_width.p90[by_width.width > 1]
    return {"batched_p90_s_lowest": float(p90.min()),
            "batched_p90_s_highest": float(p90.max()),
            "requests_per_width": dict(zip(by_width.width.astype(str), by_width.requests.astype(int)))}


def calls_per_request(calls: pandas.DataFrame, output_directory: pathlib.Path) -> dict:
    """Per category, the mean number of calls of the requests whose first call is of it."""
    means = tools.width_by_category(calls)
    positions = list(range(len(means)))[::-1]
    colors = [style.SHELL_TOOL_COLOR if category.startswith("*") else style.NATIVE_TOOL_COLOR for category in means.index]
    figure, plot = style.new_figure()
    plot.barh(positions, [mean - 1.0 for mean in means], left=1.0, height=0.8, color=colors, lw=0, zorder=3)
    axes.category_axis(plot, "y", positions, list(means.index))
    plot.tick_params(axis="y", labelsize=style.FONT_CATEGORY * 0.8)  # fifteen rows fit the standard panel at this size
    plot.set_ylim(-0.7, len(means) - 0.3)
    axes.apply(plot, "x", "mean_calls", "calls", low=1.0, high=max(1.5, float(means.max()) * 1.08))
    plot.grid(False, axis="y", which="both")
    style.save(figure, output_directory, "calls_per_request_by_category")
    return {"mean_calls_per_request": {category: float(mean) for category, mean in means.items()}}
