from src.evaluation.scores.accuracy import AccuracyScores
from src.evaluation.scores.conformance import ConformanceScores
from src.evaluation.scores.distribution import MIN_REFERENCE_OCCURRENCES, DistributionScores
from src.registry import Registry
from src.scalar_metrics import Metric

# Score families in report order.
FAMILIES = (AccuracyScores, ConformanceScores, DistributionScores)

# Distributional metrics aggregate only comparable prefixes.
COMPARABLE_METRICS = frozenset(entry.key for entry in DistributionScores.metrics())


def _declared() -> dict[str, Metric]:
    """Collect score declarations, rejecting duplicate names.

    Returns:
        Metric declarations keyed by field name.

    Raises:
        ValueError: If score families reuse a field name.
    """
    entries: dict[str, Metric] = {}
    for family in FAMILIES:
        for declaration in family.metrics():
            if declaration.key in entries:
                raise ValueError(
                    f'{declaration.key} is declared twice. A report holds every score in one '
                    f'namespace, so a name belongs to a single field.'
                )
            entries[declaration.key] = declaration
    return entries


# Score metadata assembled from the declared family fields.
METRICS = Registry[Metric](
    kind='metric',
    where='the score fields of the families in src/evaluation/scores/',
    entries=_declared(),
)

__all__ = [
    'COMPARABLE_METRICS',
    'FAMILIES',
    'METRICS',
    'MIN_REFERENCE_OCCURRENCES',
    'AccuracyScores',
    'ConformanceScores',
    'DistributionScores',
]
