from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from typing import Self

from src.evaluation.scores import FAMILIES, AccuracyScores, ConformanceScores, DistributionScores
from src.inference.generation import Generation
from src.logs import ContinuationIndex
from src.logs.declare import ConformanceChecker


@dataclass(frozen=True, slots=True)
class PrefixSummary:
    """Scores for one prefix, returned by a worker."""

    prefix_len: int
    suffix_len: int
    accuracy: AccuracyScores
    conformance: ConformanceScores
    distribution: DistributionScores

    @classmethod
    def of(
        cls,
        generation: Generation,
        *,
        checker: ConformanceChecker,
        index: ContinuationIndex,
    ) -> Self:
        """Score one generated suffix against truth, constraints, and observed continuations.

        Args:
            generation: Decoded model output for one prefix.
            checker: Process-constraint checker.
            index: Observed continuations by prefix.

        Returns:
            Scores for the prefix.
        """
        return cls(
            prefix_len=generation.prefix_len,
            suffix_len=len(generation.truth),
            accuracy=AccuracyScores.of(generation),
            conformance=ConformanceScores.of(generation, checker=checker),
            distribution=DistributionScores.of(generation, index=index),
        )


def _distribution(prefixes: Sequence[PrefixSummary]) -> tuple[int, DistributionScores]:
    """Average distributional scores over comparable prefixes only.

    Args:
        prefixes: Prefix summaries to aggregate.

    Returns:
        Comparable-prefix count and its mean scores.
    """
    comparable = [prefix.distribution for prefix in prefixes if prefix.distribution.comparable]
    if not comparable:
        return 0, DistributionScores.undefined()
    return len(comparable), DistributionScores.mean(comparable)


@dataclass(frozen=True)
class LengthSummary:
    """Mean scores for prefixes with one shared length."""

    length: int
    prefixes: int
    # Comparable prefixes used for distributional metrics.
    compared: int
    accuracy: AccuracyScores
    conformance: ConformanceScores
    distribution: DistributionScores

    @classmethod
    def of(cls, prefixes: Sequence[PrefixSummary], *, length: int) -> Self:
        """Aggregate scores for prefixes with a common length.

        Args:
            prefixes: Prefix summaries sharing the length.
            length: Shared prefix or suffix length.

        Returns:
            Aggregate scores for the length.
        """
        compared, distribution = _distribution(prefixes)
        return cls(
            length=length,
            prefixes=len(prefixes),
            compared=compared,
            accuracy=AccuracyScores.mean([prefix.accuracy for prefix in prefixes]),
            conformance=ConformanceScores.mean([prefix.conformance for prefix in prefixes]),
            distribution=distribution,
        )


def _by_length(buckets: dict[int, list[PrefixSummary]]) -> list[LengthSummary]:
    """Summarize length buckets in ascending order.

    Args:
        buckets: Prefix summaries grouped by length.

    Returns:
        One aggregate per length.
    """
    return [LengthSummary.of(buckets[length], length=length) for length in sorted(buckets)]


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate evaluation scores for one run."""

    prefixes: int
    # Comparable prefixes used for distributional metrics.
    compared: int
    accuracy: AccuracyScores
    conformance: ConformanceScores
    distribution: DistributionScores
    # Sorted by prefix length.
    by_prefix_length: list[LengthSummary]
    # Sorted by ground-truth suffix length.
    by_suffix_length: list[LengthSummary]

    @classmethod
    def of(cls, prefixes: Iterable[PrefixSummary]) -> Self:
        """Aggregate prefix scores overall and by prefix and suffix length.

        Args:
            prefixes: Prefix summaries to aggregate.

        Returns:
            Overall and length-bucketed evaluation scores.
        """
        prefix_buckets: dict[int, list[PrefixSummary]] = {}
        suffix_buckets: dict[int, list[PrefixSummary]] = {}

        for prefix in prefixes:
            # Group by both reported length axes.
            prefix_buckets.setdefault(prefix.prefix_len, []).append(prefix)
            suffix_buckets.setdefault(prefix.suffix_len, []).append(prefix)

        # Recover all prefixes for the overall aggregate.
        every_prefix = [prefix for bucket in prefix_buckets.values() for prefix in bucket]
        compared, distribution = _distribution(every_prefix)
        return cls(
            prefixes=len(every_prefix),
            compared=compared,
            accuracy=AccuracyScores.mean([prefix.accuracy for prefix in every_prefix]),
            conformance=ConformanceScores.mean([prefix.conformance for prefix in every_prefix]),
            distribution=distribution,
            by_prefix_length=_by_length(prefix_buckets),
            by_suffix_length=_by_length(suffix_buckets),
        )


# Types that expose all three score families.
type Summarized = PrefixSummary | LengthSummary | EvaluationSummary

# Cache each family's metric fields.
_FIELD_NAMES = {family: tuple(entry.name for entry in fields(family)) for family in FAMILIES}


def flatten_scores(summary: Summarized) -> dict[str, float]:
    """Flatten a summary's score families into a metric mapping.

    Args:
        summary: Prefix, length, or evaluation aggregate.

    Returns:
        Metric values keyed by metric name.
    """
    return {
        name: getattr(family, name)
        for family in (
            summary.accuracy,
            summary.conformance,
            summary.distribution,
        )
        for name in _FIELD_NAMES[type(family)]
    }
