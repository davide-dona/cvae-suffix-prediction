from dataclasses import dataclass

from src.evaluation import Axis
from src.evaluation.scores import METRICS
from src.scalar_metrics import Direction
from src.visualization.catalogue.entry import MetricEntry


@dataclass(frozen=True)
class ColumnGroup:
    """A contiguous group of table columns sharing one header."""

    label: str
    span: int


@dataclass(frozen=True)
class Table:
    """Definition of one comparison table and its columns."""

    name: str
    axis: Axis
    note: str
    columns: tuple[MetricEntry, ...]
    column_groups: tuple[ColumnGroup, ...] = ()

    def __post_init__(self) -> None:
        undirected = [
            entry.metric.key for entry in self.columns if entry.metric.direction is Direction.NONE
        ]
        if undirected:
            raise ValueError(
                f'the {self.name} table holds {", ".join(undirected)}, which have no better value '
                f'and so no cell a reader could rank or the emphasis could mark. A property of the '
                f"log is drawn in FIGURES as the log's own series rather than tabulated."
            )
        if any(group.span < 1 for group in self.column_groups):
            raise ValueError(f'every {self.name} table column group must span at least one column.')
        if self.column_groups and sum(group.span for group in self.column_groups) != len(
            self.columns
        ):
            raise ValueError(
                f'the {self.name} table groups {sum(group.span for group in self.column_groups)} '
                f'columns but declares {len(self.columns)}.'
            )


# Each table answers one evaluation question with directional metrics.
TABLES = (
    # Point prediction against the observed suffix.
    Table(
        name='accuracy-point',
        axis=Axis.OVERALL,
        note='Mean absolute error per prefix; length in events, times in days.',
        columns=(
            MetricEntry(METRICS['dls_point'], 'DLS'),
            MetricEntry(METRICS['length_ae_point'], 'Length'),
            MetricEntry(METRICS['remaining_time_ae_point_days'], 'Rem. time'),
            MetricEntry(METRICS['cycle_time_ae_point_days'], 'Event time'),
        ),
    ),
    # Distributional fidelity to observed continuations.
    Table(
        name='fidelity',
        axis=Axis.OVERALL,
        note='W1 is the 1-Wasserstein distance; length in events, times in days.',
        columns=(
            MetricEntry(METRICS['emsc'], 'EMSC'),
            MetricEntry(METRICS['continuation_precision'], 'Precision'),
            MetricEntry(METRICS['continuation_recall'], 'Recall'),
            MetricEntry(METRICS['length_wasserstein'], 'Length W1'),
            MetricEntry(METRICS['remaining_time_wasserstein_days'], 'Rem. time W1'),
            MetricEntry(METRICS['activity_time_wasserstein_days'], 'Event time W1'),
        ),
    ),
    # Sample quality against the observed suffix.
    Table(
        name='accuracy-generative',
        axis=Axis.OVERALL,
        note='Hit rate@k is the share of prefixes whose first k samples include the exact ground '
        'truth suffix, @any the share whose samples include it at all, and hit share the share of '
        "a prefix's samples that were exactly it. CRPS is the continuous ranked probability score "
        'of the samples; length in events, times in days.',
        columns=(
            MetricEntry(METRICS['hit_rate_at_1'], 'Hit rate@1'),
            MetricEntry(METRICS['hit_rate_at_5'], 'Hit rate@5'),
            MetricEntry(METRICS['hit_rate_at_10'], 'Hit rate@10'),
            MetricEntry(METRICS['hit_rate_any'], 'Hit rate@any'),
            MetricEntry(METRICS['hit_share'], 'Hit share'),
            MetricEntry(METRICS['length_crps'], 'Length CRPS'),
            MetricEntry(METRICS['remaining_time_crps_days'], 'Rem. time CRPS'),
            MetricEntry(METRICS['cycle_time_crps_days'], 'Event time CRPS'),
        ),
    ),
    # Calibration gaps at three central-interval levels.
    Table(
        name='calibration',
        axis=Axis.OVERALL,
        note="Each cell is the empirical coverage of the samples' central interval less the level "
        'it covers, so 0 is calibrated, negative over-confident and positive over-dispersed. An '
        'interval read off a finite sample is narrow, which costs a calibrated model a point or '
        'two on every column.',
        columns=(
            MetricEntry(METRICS['length_coverage_gap_50'], r'50\%'),
            MetricEntry(METRICS['length_coverage_gap_75'], r'75\%'),
            MetricEntry(METRICS['length_coverage_gap_95'], r'95\%'),
            MetricEntry(METRICS['remaining_time_coverage_gap_50'], r'50\%'),
            MetricEntry(METRICS['remaining_time_coverage_gap_75'], r'75\%'),
            MetricEntry(METRICS['remaining_time_coverage_gap_95'], r'95\%'),
            MetricEntry(METRICS['cycle_time_coverage_gap_50'], r'50\%'),
            MetricEntry(METRICS['cycle_time_coverage_gap_75'], r'75\%'),
            MetricEntry(METRICS['cycle_time_coverage_gap_95'], r'95\%'),
        ),
        column_groups=(
            ColumnGroup('Length', 3),
            ColumnGroup('Remaining time', 3),
            ColumnGroup('Event time', 3),
        ),
    ),
)
