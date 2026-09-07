import logging
from collections.abc import Sequence

import matplotlib as mpl
from matplotlib.artist import Artist
from matplotlib.figure import Figure

# IEEE conference text widths in inches.
COLUMN_WIDTH = 3.487  # (43pc - 1pc) / 2 = 252pt
PAGE_WIDTH = 7.140  # 43pc = 516pt
# Figure height-to-width ratio.
ASPECT = 0.618

# Height reserved for the shared legend.
LEGEND_HEIGHT = 0.40
# Title wrapping width and maximum x-axis bins.
TITLE_WIDTH = 22
PANEL_X_BINS = 5
# Maximum visible markers per series.
MAX_MARKERS = 8
# Upper-bound headroom as a share of its range.
Y_HEADROOM = 0.05

# Shared Matplotlib settings.
_PAPER_RC = {
    # Use the first available Times-compatible font.
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Times', 'STIX Two Text', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.titlesize': 9,
    'figure.labelsize': 8,
    # Keep only data-bearing spines.
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'axes.labelpad': 3.0,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'axes.grid': False,
    'lines.linewidth': 1.0,
    'lines.markersize': 2.4,
    'lines.markeredgewidth': 0.3,
    'legend.frameon': False,
    'legend.borderpad': 0.0,
    'legend.handlelength': 1.8,
    'legend.columnspacing': 0.9,
    # Resolution for rasterized output.
    'savefig.dpi': 400,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.01,
    # Keep PDF text selectable.
    'pdf.fonttype': 42,
}


def apply_style() -> None:
    """Apply shared non-interactive figure styling.

    Returns:
        None.
    """
    # Suppress noisy third-party font logging.
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    # Figures are written without a display.
    mpl.use('Agg')
    mpl.rcParams.update(_PAPER_RC)


def legend_above(figure: Figure, handles: Sequence[Artist], keys: Sequence[str]) -> None:
    """Draw a shared legend above the panels.

    Args:
        figure: Figure containing the panels.
        handles: Artists represented by the legend.
        keys: Labels for the artists.
    """
    figure.legend(handles, keys, loc='outside upper center', ncols=len(keys))
