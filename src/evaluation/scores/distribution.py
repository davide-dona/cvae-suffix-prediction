from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Self

import numpy as np
import ot
from scipy.stats import wasserstein_distance

from src.evaluation.scores.accuracy import MINUTES_PER_DAY
from src.inference.generation import Generation
from src.logs import ContinuationIndex, Continuations
from src.scalar_metrics import Direction, Owner, ScalarMetrics, Unit, metric
from src.suffixes import distances, diversity

# Minimum observations required for a meaningful reference distribution.
MIN_REFERENCE_OCCURRENCES = 5


@dataclass(frozen=True, slots=True)
class DistributionScores(ScalarMetrics):
    """Similarity between generated and observed continuation distributions."""

    # Earth Movers' Stochastic Conformance similarity.
    emsc: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)

    # Recall is occurrence-weighted; precision is draw-weighted.
    continuation_recall: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    continuation_precision: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)

    # Wasserstein distances for length, remaining time, and activity timing.
    length_wasserstein: float = metric(unit=Unit.EVENTS, direction=Direction.LOWER)
    remaining_time_wasserstein_days: float = metric(unit=Unit.DAYS, direction=Direction.LOWER)
    activity_time_wasserstein_days: float = metric(unit=Unit.DAYS, direction=Direction.LOWER)

    # Generated diversity and unique-sequence rate.
    sample_diversity: float = metric(unit=Unit.SHARE)
    unique_sample_rate: float = metric(unit=Unit.SHARE)

    # Observed-continuation diversity, owned by the log.
    reference_diversity: float = metric(unit=Unit.SHARE, owner=Owner.LOG)

    # Number of distinct observed continuations.
    reference_size: float = metric(unit=Unit.COUNT, owner=Owner.LOG)

    # Number of observed occurrences of this prefix.
    reference_occurrences: float = metric(unit=Unit.COUNT, owner=Owner.LOG)

    @classmethod
    def of(cls, generation: Generation, *, index: ContinuationIndex) -> Self:
        """Score a generation against continuations observed for its prefix.

        Args:
            generation: Decoded model output for one prefix.
            index: Observed continuations by prefix.

        Returns:
            Distributional scores for the prefix.
        """
        samples = generation.samples
        references = index.continuations(generation.prefix_activities)

        # Distinct generated suffixes and their draw counts.
        suffixes, counts = samples.suffixes, samples.counts
        draws = len(samples)

        # Draw counts weight generated diversity.
        sample_diversity = diversity(suffixes, weights=counts)

        observed, generated = set(references.suffixes), set(suffixes)
        covered = sum(
            weight
            for suffix, weight in zip(references.suffixes, references.weights, strict=True)
            if suffix in generated
        )

        lengths = [float(len(suffix)) for suffix in suffixes] or [0.0]
        remaining = [events.remaining_time_minutes for events in samples.events] or [0.0]

        # Pool cycle times by preceding activity across draws.
        drawn: dict[str, list[float]] = {}
        for events in samples.events:
            for activity, cycle_time in zip(
                events.activities, events.cycle_time_minutes, strict=True
            ):
                drawn.setdefault(activity, []).append(cycle_time)

        return cls(
            emsc=emsc(suffixes=suffixes, counts=counts, references=references),
            continuation_recall=covered / references.occurrences,
            continuation_precision=(
                float(counts @ [float(suffix in observed) for suffix in suffixes]) / draws
                if len(suffixes) and draws
                else 0.0
            ),
            # Weights recover the length distribution without expanding draws.
            length_wasserstein=wasserstein_distance(
                u_values=lengths,
                u_weights=counts if len(suffixes) else None,
                v_values=[float(len(suffix)) for suffix in references.suffixes],
                v_weights=references.weights,
            ),
            remaining_time_wasserstein_days=wasserstein_distance(
                u_values=remaining,
                v_values=references.remaining_times,
            )
            / MINUTES_PER_DAY,
            activity_time_wasserstein_days=activity_time_wasserstein_minutes(
                generated=drawn, observed=references.cycle_times
            )
            / MINUTES_PER_DAY,
            sample_diversity=sample_diversity,
            unique_sample_rate=len(suffixes) / draws if draws else 0.0,
            reference_diversity=references.diversity,
            reference_size=float(len(references.suffixes)),
            reference_occurrences=references.occurrences,
        )

    @property
    def comparable(self) -> bool:
        """Return whether the reference has enough observations for comparison.

        Returns:
            Whether the prefix meets `MIN_REFERENCE_OCCURRENCES`.
        """
        return self.reference_occurrences >= MIN_REFERENCE_OCCURRENCES

    @classmethod
    def undefined(cls) -> Self:
        """Return NaN scores for an aggregate with no comparable prefixes.

        Returns:
            A score instance whose fields are all NaN.
        """
        return cls(**{entry.name: float('nan') for entry in fields(cls)})


def emsc(suffixes: tuple[str, ...], counts: np.ndarray, references: Continuations) -> float:
    """Return optimal-transport similarity of generated and observed suffixes.

    Args:
        suffixes: Distinct generated suffixes.
        counts: Draw count for each generated suffix.
        references: Observed continuations for the prefix.

    Returns:
        Earth Movers' Stochastic Conformance similarity in `[0, 1]`.
    """
    if not suffixes:
        return 0.0

    # Rows are generated suffixes; columns are observed continuations.
    cost = distances(queries=suffixes, choices=references.suffixes, dtype=np.float64)
    drawn = counts / counts.sum()
    weights = references.weights / references.occurrences
    return 1.0 - float(ot.emd2(a=drawn, b=weights, M=cost))


def activity_time_wasserstein_minutes(
    generated: Mapping[str, Sequence[float]], observed: Mapping[str, np.ndarray]
) -> float:
    """Return occurrence-weighted timing Wasserstein distance by activity.

    Args:
        generated: Cycle times grouped by generated activity.
        observed: Cycle times grouped by observed activity.

    Returns:
        Mean 1-Wasserstein distance in minutes.
    """
    scored = [
        (
            float(len(cycle_times)),
            wasserstein_distance(u_values=generated[activity], v_values=cycle_times),
        )
        for activity, cycle_times in observed.items()
        if activity in generated
    ]
    if scored:
        return sum(weight * distance for weight, distance in scored) / sum(
            weight for weight, _ in scored
        )
    return wasserstein_distance(
        u_values=[cycle_time for cycle_times in generated.values() for cycle_time in cycle_times]
        or [0.0],
        v_values=[cycle_time for cycle_times in observed.values() for cycle_time in cycle_times],
    )
