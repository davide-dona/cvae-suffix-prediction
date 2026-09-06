from src.evaluation.scores.accuracy import AccuracyScores
from src.evaluation.scores.conformance import ConformanceScores
from src.evaluation.scores.distribution import MIN_REFERENCE_OCCURRENCES, DistributionScores
from src.registry import Registry
from src.scalar_metrics import Metric

# The families a report carries, in the order it lays them out.
FAMILIES = (AccuracyScores, ConformanceScores, DistributionScores)

# Which of a report's numbers are read over the prefixes `DistributionScores.comparable`
# admits rather than over every prefix scored. The whole of that family and nothing else:
# each of its scores compares the draws against the continuations one prefix was observed to
# take, and none of them is a comparison at all where those are a sample of one. Every reader
# that reduces per-prefix rows to a mean, a band or a p-value splits its metrics on this.
COMPARABLE_METRICS = frozenset(entry.key for entry in DistributionScores.metrics())


def _declared() -> dict[str, Metric]:
    """Every number a report carries, keyed by the field it was declared on.

    Returns:
        The declarations of every family in one namespace, exactly as `summary.flatten_scores`
        merges them.
    Raises:
        ValueError: If two families declare the same name, which would leave a figure asking for
            one and reading whichever was flattened last.
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


# Every number a report carries, which is every value the `metric` column of `read_reports` can
# hold. Assembled from the score fields themselves, so a score added to a family is a name a
# figure can spell with no change here, and a score renamed there is renamed everywhere at once.
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
