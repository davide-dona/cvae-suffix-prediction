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

# X-axis labels by length breakdown.
AXIS_LABELS = {Axis.PREFIX: 'Prefix length', Axis.SUFFIX: 'Suffix length'}


def _draw_metric(axes: Axes, frame: pd.DataFrame, entry: MetricEntry) -> int:
    """Draw one metric and its confidence bands.

    Args:
        axes: Panel to draw on.
        frame: Report rows for one dataset and breakdown with intervals.
        entry: Metric display definition.

    Returns:
        Longest reported length.
    """
    # Select rows for the requested metric.
    values = frame[frame['metric'] == entry.key]
    drawn = labels.MODELS.ordered(values['model'])
    # Log-owned metrics produce one shared series.
    if entry.metric.owner is Owner.LOG:
        series = [(model, labels.LOG_STYLE) for model in drawn[:1]]
    else:
        series = [(model, labels.MODELS[model]) for model in drawn]

    lines = [
        (values[values['model'] == model].sort_values('length'), style) for model, style in series
    ]

    # Draw bands first so lines remain visible.
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
    """Draw a panel and return its longest reported length.

    Args:
        axes: Panel to draw on.
        frame: Report rows for one dataset and breakdown with intervals.
        panel: Metrics displayed together.
        x_bins: Maximum x-axis tick bins.

    Returns:
        Longest reported length across the panel.
    """
    longest = max(_draw_metric(axes, frame, entry) for entry in panel)
    axes.xaxis.set_major_locator(MaxNLocator(nbins=x_bins, integer=True))
    # Unbounded sides use Matplotlib's automatic limits.
    bottom, top = panel[0].bounds
    if top is not None:
        # Keep values at the bound from appearing clipped.
        top += Y_HEADROOM * (top - (bottom if bottom is not None else 0.0))
    axes.set_ylim(bottom=bottom, top=top)
    return longest


def _link_x_axes(grid: np.ndarray, breakdowns: list[Axis], longest: list[list[int]]) -> None:
    """Share x-axis limits within each dataset and breakdown.

    Args:
        grid: Figure axes indexed by row and dataset.
        breakdowns: Breakdown represented by each row.
        longest: Longest reported length for each panel.
    """
    for column in range(grid.shape[1]):
        for breakdown in dict.fromkeys(breakdowns):
            rows = [row for row, drawn in enumerate(breakdowns) if drawn == breakdown]
            # Align panel limits to the longest observed series.
            right = max(longest[row][column] for row in rows)
            for row in rows:
                grid[row][column].set_xlim(left=1, right=right)
            # Show x tick labels only on the lowest linked panel.
            for row in rows[:-1]:
                grid[row][column].tick_params(labelbottom=False)


def compose_figure(frame: pd.DataFrame, plot: Plot) -> Figure:
    """Compose a catalogue figure across all datasets.

    Args:
        frame: Report rows with confidence intervals.
        plot: Figure definition.

    Returns:
        Composed Matplotlib figure.
    """
    datasets = labels.DATASETS.ordered(frame['dataset'])
    # Group panels by breakdown to share their x-axis.
    rows = [(breakdown, panel) for breakdown in plot.breakdowns for panel in plot.panels]
    # Preserve column width until the figure reaches page width.
    width = min(PAGE_WIDTH, len(datasets) * COLUMN_WIDTH)
    figure, grid = plt.subplots(
        nrows=len(rows),
        ncols=len(datasets),
        figsize=(width, len(rows) * width / len(datasets) * ASPECT + LEGEND_HEIGHT),
        squeeze=False,
        constrained_layout=True,
    )
    # Track panel extents for linked x-axes.
    longest = []
    for (breakdown, panel), row in zip(rows, grid, strict=True):
        drawn = frame[frame['axis'] == breakdown]
        longest.append(
            [
                _draw_panel(axes, drawn[drawn['dataset'] == dataset], panel, x_bins=PANEL_X_BINS)
                for axes, dataset in zip(row, datasets, strict=True)
            ]
        )
        # The primary metric labels the row.
        entry = panel[0]
        row[0].set_ylabel(textwrap.fill(entry.axis_label, width=TITLE_WIDTH))
        if entry.shares_scale:
            # Fixed-scale rows need y tick labels once.
            for axes in row[1:]:
                axes.tick_params(labelleft=False)

    _link_x_axes(grid, [breakdown for breakdown, _ in rows], longest)

    # Each column represents one dataset.
    for axes, dataset in zip(grid[0], datasets, strict=True):
        axes.set_title(labels.DATASETS[dataset])

    if len(plot.breakdowns) == 1:
        figure.supxlabel(AXIS_LABELS[plot.breakdowns[0]])
    else:
        # Label each breakdown below its final panel row.
        for breakdown in plot.breakdowns:
            foot = max(row for row, (drawn, _) in enumerate(rows) if drawn == breakdown)
            for axes in grid[foot]:
                axes.set_xlabel(AXIS_LABELS[breakdown])

    # Deduplicate legend entries across panels.
    keys: dict[str, Artist] = {}
    for row in grid:
        for axes in row:
            handles, written = axes.get_legend_handles_labels()
            for label, handle in zip(written, handles, strict=True):
                keys.setdefault(label, handle)
    if len(keys) > 1:
        legend_above(figure, list(keys.values()), list(keys))
    return figure
