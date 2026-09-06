import math
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from src.evaluation import Axis
from src.scalar_metrics import Owner
from src.visualization import labels
from src.visualization.catalogue import MetricEntry, Plot
from src.visualization.style import (
    ASPECT,
    BAND_ALPHA,
    BAND_Z,
    COLUMN_WIDTH,
    LEGEND_HEIGHT,
    LINE_Z,
    MAX_MARKERS,
    PAGE_WIDTH,
    PANEL_X_BINS,
    TITLE_WIDTH,
    Y_HEADROOM,
    legend_above,
)

# What the x-axis of a figure drawn against each length breakdown is called
AXIS_LABELS = {Axis.PREFIX: 'Prefix length', Axis.SUFFIX: 'Suffix length'}


def _draw_metric(axes: Axes, frame: pd.DataFrame, entry: MetricEntry) -> int:
    """Draw one metric onto one set of axes over one log's rows: a line per model, or a single line
    in the log's own style where the metric is the log's rather than a model's. Every line carries
    the confidence interval its length's mean is pinned down to.

    Args:
        axes: The panel to draw onto.
        frame: The rows of one log and one breakdown, from `read_reports` with the bounds of
            `read_intervals` joined onto it. A row whose bounds are null keeps its line and loses
            only its band.
        entry: The metric to draw, and what this figure calls it.
    Returns:
        The longest length any model reports here, so the panels drawn over the same lengths can
        end where the longest of them does.
    """
    # Retrieve the rows of one metric
    values = frame[frame['metric'] == entry.key]
    drawn = labels.MODELS.ordered(values['model'])
    # A metric the log owns is the same number for every model of that log, so it is drawn once in
    # the log's own style rather than as one line per model in each model's colour, which is as
    # many coincident lines as there are runs.
    if entry.metric.owner is Owner.LOG:
        series = [(model, labels.LOG_STYLE) for model in drawn[:1]]
    else:
        series = [(model, labels.MODELS[model]) for model in drawn]

    lines = [
        (values[values['model'] == model].sort_values('length'), style) for model, style in series
    ]

    # Every band first, so the last series drawn never covers the first series' line. A band is the
    # interval that length's mean could be off by, which is what settles whether two lines are
    # really apart at a length and whether a model's line is really below the log's own.
    for line, style in lines:
        bounded = line.dropna(subset=['low', 'high'])
        axes.fill_between(
            bounded['length'],
            bounded['low'],
            bounded['high'],
            color=style.color,
            alpha=BAND_ALPHA,
            linewidth=0.0,
            zorder=BAND_Z,
        )

    longest = 1
    for line, style in lines:
        longest = max(longest, int(line['length'].max()))
        axes.plot(
            line['length'],
            line['value'],
            label=style.label,
            color=style.color,
            marker=style.marker,
            linestyle=style.linestyle,
            markevery=max(1, math.ceil(len(line) / MAX_MARKERS)),
            zorder=LINE_Z,
        )
    return longest


def _draw_panel(
    axes: Axes, frame: pd.DataFrame, panel: tuple[MetricEntry, ...], *, x_bins: int | str
) -> int:
    """Draw every metric of one panel onto one set of axes over one log's rows.

    Args:
        axes: The panel to draw onto.
        frame: The rows of one log and one breakdown, from `read_reports` with the bounds of
            `read_intervals` joined onto it.
        panel: The metrics to draw together, e.g. a model's estimator with the log's own value
            beside it. The first names the panel's axis and its limits; the rest share both, which
            holds for every panel the catalogue declares since a target is read on its estimator's
            own scale.
        x_bins: How many ticks the x-axis is allowed.
    Returns:
        The longest length any line of the panel reports, so the panels drawn over the same
        lengths can end where the longest of them does.
    """
    longest = max(_draw_metric(axes, frame, entry) for entry in panel)
    axes.xaxis.set_major_locator(MaxNLocator(nbins=x_bins, integer=True))
    # Either end the metric leaves open is left to the data, matplotlib scaling it as it would.
    bottom, top = panel[0].bounds
    if top is not None:
        # Headroom past the bound itself, so a line approaching it does not read as clipped.
        top += Y_HEADROOM * (top - (bottom if bottom is not None else 0.0))
    axes.set_ylim(bottom=bottom, top=top)
    return longest


def _link_x_axes(grid: np.ndarray, breakdowns: list[Axis], longest: list[list[int]]) -> None:
    """Put the panels of one column drawn against one breakdown on one x-axis.

    By column and breakdown, not by row: a column is one log, and two logs are two processes whose
    lengths are not on one scale, while a prefix length and a suffix length are two different
    quantities of the one log.

    Args:
        grid: The figure's panels, a row per metric and breakdown and a column per log.
        breakdowns: The breakdown each row of the grid is drawn against.
        longest: The longest length drawn in each panel, from `_draw_panel`.
    """
    for column in range(grid.shape[1]):
        for breakdown in dict.fromkeys(breakdowns):
            rows = [row for row, drawn in enumerate(breakdowns) if drawn == breakdown]
            # The axis ends where the data does rather than at the padding matplotlib would leave
            # past it, and every panel over these lengths ends there together.
            right = max(longest[row][column] for row in rows)
            for row in rows:
                grid[row][column].set_xlim(left=1, right=right)
            # The lowest of them prints the lengths for all of them, the rest being the same ticks
            # in the height of a panel.
            for row in rows[:-1]:
                grid[row][column].tick_params(labelbottom=False)


def compose_figure(frame: pd.DataFrame, plot: Plot) -> Figure:
    """Compose one figure of the catalogue, covering every log at once, untitled since a paper
    captions its figures. A column per log and a row per metric and length breakdown.

    Args:
        frame: Every report's rows the figure is drawn from, from `read_reports`.
        plot: Which figure to draw.
    Returns:
        The finished figure.
    """
    datasets = labels.DATASETS.ordered(frame['dataset'])
    # By breakdown first, so the rows drawn over one set of lengths are a block: they share their
    # x-axis, print its ticks once at the foot of the block and are named by it there.
    rows = [(breakdown, panel) for breakdown in plot.breakdowns for panel in plot.panels]
    # Page width once there are enough logs to fill it, and a column of panels of the usual width
    # below that, so drawing one or two logs across gives a figure of the size the same panels have
    # everywhere else rather than one panel blown up to the width of the page.
    width = min(PAGE_WIDTH, len(datasets) * COLUMN_WIDTH)
    figure, grid = plt.subplots(
        nrows=len(rows),
        ncols=len(datasets),
        figsize=(width, len(rows) * width / len(datasets) * ASPECT + LEGEND_HEIGHT),
        squeeze=False,
        constrained_layout=True,
    )
    # What each panel drew, since the x-axis a block of rows ends at is the longest of them.
    longest = []
    for (breakdown, panel), row in zip(rows, grid, strict=True):
        drawn = frame[frame['axis'] == breakdown]
        longest.append(
            [
                _draw_panel(axes, drawn[drawn['dataset'] == dataset], panel, x_bins=PANEL_X_BINS)
                for axes, dataset in zip(row, datasets, strict=True)
            ]
        )
        # The panel's primary metric names the row it is drawn along, its unit included, the way
        # it names the y-axis of a single-panel figure.
        # Wrapped to the width a panel title is: a row is only as tall as one panel, and a metric
        # whose name and unit run past that would otherwise be set taller than what it labels.
        entry = panel[0]
        row[0].set_ylabel(textwrap.fill(entry.axis_label, width=TITLE_WIDTH))
        if entry.shares_scale:
            # A row drawn over one fixed range in every panel has ticks that are the metric's and
            # not each log's: printing them once says the same thing in the width of one panel
            # less.
            for axes in row[1:]:
                axes.tick_params(labelleft=False)

    _link_x_axes(grid, [breakdown for breakdown, _ in rows], longest)

    # The logs title the top row alone: the column below a title is one log throughout.
    for axes, dataset in zip(grid[0], datasets, strict=True):
        axes.set_title(labels.DATASETS[dataset])

    if len(plot.breakdowns) == 1:
        figure.supxlabel(AXIS_LABELS[plot.breakdowns[0]])
    else:
        # One label under the whole figure could not name two breakdowns, so each block is named
        # at its own foot, under the panels whose lengths it is naming.
        for breakdown in plot.breakdowns:
            foot = max(row for row, (drawn, _) in enumerate(rows) if drawn == breakdown)
            for axes in grid[foot]:
                axes.set_xlabel(AXIS_LABELS[breakdown])

    # Gathered over every panel rather than off the first: a row of a log's own metric draws one
    # series the model rows do not, so a legend read off one panel would leave it unnamed. Keyed by
    # label, so the models repeated down the rows contribute one key each.
    keys: dict[str, Artist] = {}
    for row in grid:
        for axes in row:
            handles, written = axes.get_legend_handles_labels()
            for label, handle in zip(written, handles, strict=True):
                keys.setdefault(label, handle)
    if len(keys) > 1:
        legend_above(figure, list(keys.values()), list(keys))
    return figure
