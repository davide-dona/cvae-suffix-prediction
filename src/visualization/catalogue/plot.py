from dataclasses import dataclass

from src.evaluation import Axis
from src.evaluation.scores import METRICS
from src.visualization.catalogue.entry import MetricEntry


@dataclass(frozen=True)
class Plot:
    """One figure of the catalogue, covering every log at once: what its file is called, which
    length breakdowns it is drawn against, and the panels it draws.

    A panel is drawn once per breakdown, so the figure holds one row per pair of the two and one
    column per log. A panel groups one or more metrics onto the same axes: an estimator together
    with the log's own value it is read against, where the process gives one, so the target a line
    is judged by sits on its panel rather than in a panel of its own. Naming both breakdowns puts
    them in one figure rather than in two, which is worth doing where the two readings say
    different things and not otherwise: they are not independent, since every prefix of a case is
    scored and so a long prefix leaves a short suffix.
    """

    name: str
    breakdowns: tuple[Axis, ...]
    panels: tuple[tuple[MetricEntry, ...], ...]


# Every figure of the catalogue.
# Each one draws a group of panels against one or both length breakdowns (PREFIX and SUFFIX).
# One column per log, one row per panel and breakdown, one line per model within a panel, plus the
# log's own line where a panel names it too.
#
# A quantity a model answers both ways is one figure with a row each rather than two files: the
# point estimate and the sample mean are read against each other, and two figures of one quantity
# carry two legends and two sets of lengths to say it.
#
# Every line drawn here carries the confidence interval `src.uncertainty.intervals` bounds that
# length's mean by, so a panel says not only where each model sits but whether the models are
# really apart there and whether one is really below the log's own line.
FIGURES = (
    Plot(
        name='dls-by-length',
        breakdowns=(Axis.SUFFIX,),
        panels=(
            (MetricEntry(METRICS['dls_point'], 'DLS (point)'),),
            (MetricEntry(METRICS['dls_mean'], 'DLS (sample mean)'),),
        ),
    ),
    Plot(
        name='conformance-by-suffix-length',
        breakdowns=(Axis.SUFFIX,),
        panels=(
            (
                MetricEntry(METRICS['conformance_point'], 'Conformance (point)'),
                MetricEntry(METRICS['conformance_truth'], 'Conformance (ground truth)'),
            ),
            (
                MetricEntry(METRICS['conformance_mean'], 'Conformance (sample mean)'),
                MetricEntry(METRICS['conformance_truth'], 'Conformance (ground truth)'),
            ),
        ),
    ),
    # The same check read as a verdict rather than as a share, in its own figure rather than as two
    # more rows above: a satisfied share and a share of traces satisfying everything are two
    # quantities on one axis, and a trace sitting a point below the log on the first can be a trace
    # the process does not admit at all.
    Plot(
        name='full-conformance-by-suffix-length',
        breakdowns=(Axis.SUFFIX,),
        panels=(
            (
                MetricEntry(METRICS['full_conformance_point'], 'Fully conformant (point)'),
                MetricEntry(METRICS['full_conformance_truth'], 'Fully conformant (ground truth)'),
            ),
            (
                MetricEntry(METRICS['full_conformance_mean'], 'Fully conformant (samples)'),
                MetricEntry(METRICS['full_conformance_truth'], 'Fully conformant (ground truth)'),
            ),
        ),
    ),
)
