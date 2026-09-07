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
        note='Exact-match rate is the share of generated samples that exactly match the observed '
        'suffix. W1 is the 1-Wasserstein distance; length in events, times in days.',
        columns=(
            MetricEntry(METRICS['emsc'], 'EMSC'),
            MetricEntry(METRICS['continuation_precision'], 'Precision'),
            MetricEntry(METRICS['continuation_recall'], 'Recall'),
            MetricEntry(METRICS['hit_share'], 'Exact-match rate'),
            MetricEntry(METRICS['length_wasserstein'], 'Length W1'),
            MetricEntry(METRICS['remaining_time_wasserstein_days'], 'Rem. time W1'),
            MetricEntry(METRICS['activity_time_wasserstein_days'], 'Event time W1'),
        ),
    ),
    # The samples themselves against the one continuation the log took.
    Table(
        name='generative-accuracy',
        axis=Axis.OVERALL,
        note="Read over every prefix scored rather than over the fidelity block's population, "
        'and against the one continuation the log took rather than against the continuation '
        "index. An energy score is E d(X, y) - 0.5 E d(X, X') over the samples, CRPS generalized "
        'off the real line, which charges the spread against the accuracy it buys where the '
        'sampled DLS beside it is won by putting every sample on one suffix. Exact is 1 for any '
        'two suffixes that are not the same and bigram is the multiset Jaccard distance over the '
        'ordered activity pairs a suffix holds; both are of negative type, so both are proper. '
        'DLS is 1 minus the normalized Damerau-Levenshtein similarity, which is not of negative '
        'type, so its column is the sample-side counterpart of the DLS beside it rather than a '
        'proper score.',
        columns=(
            MetricEntry(METRICS['dls_mean'], 'DLS (sample mean)'),
            MetricEntry(METRICS['energy_score_dls'], 'ES (DLS)'),
            MetricEntry(METRICS['energy_score_exact'], 'ES (exact)'),
            MetricEntry(METRICS['energy_score_bigram'], 'ES (bigram)'),
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
