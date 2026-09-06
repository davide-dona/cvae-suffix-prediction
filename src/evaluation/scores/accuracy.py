from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain, islice, repeat
from typing import Self

from src.inference.generation import Draws, Generation
from src.scalar_metrics import Direction, Owner, ScalarMetrics, Unit, mean, metric
from src.suffixes import sequence_similarity

MINUTES_PER_DAY = 1440.0


@dataclass(frozen=True, slots=True)
class AccuracyScores(ScalarMetrics):
    """How close a prefix's generated suffixes are to the one that actually followed it.

    The scores of one prefix's generated samples, or their mean over a set of prefixes.
    """

    # The Damerau-Levenshtein Similarity (DLS) between the samples and the ground truth.
    # The mean similarity of a prefix's samples to the ground truth
    dls_mean: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    # z = mean(p(z | prefix), a single greedy answer, scored against the ground truth
    dls_point: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    # The closest of a prefix's samples to the ground truth.
    dls_best: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)

    # The share of prefixes whose true suffix is exactly among their first k samples.
    hit_rate_at_1: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    hit_rate_at_5: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    hit_rate_at_10: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)

    # Absolute error (AE) between the predicted and true remaining cycle time, in days.
    remaining_time_ae_mean_days: float = metric(unit=Unit.DAYS, direction=Direction.LOWER)
    remaining_time_ae_point_days: float = metric(unit=Unit.DAYS, direction=Direction.LOWER)

    # Absolute error (AE) between the predicted and true minutes until each generated event,
    # averaged over the positions the true suffix covers, in days.
    cycle_time_ae_mean_days: float = metric(unit=Unit.DAYS, direction=Direction.LOWER)
    cycle_time_ae_point_days: float = metric(unit=Unit.DAYS, direction=Direction.LOWER)

    # Absolute error (AE) between the predicted and true suffix length, in events.
    length_ae_mean: float = metric(unit=Unit.EVENTS, direction=Direction.LOWER)
    length_ae_point: float = metric(unit=Unit.EVENTS, direction=Direction.LOWER)

    # Events left after the cut point: the scale every error above is read against. A property of
    # the prefixes scored rather than of the model, so it is flat across a training run.
    suffix_length: float = metric(unit=Unit.EVENTS, owner=Owner.LOG)

    @classmethod
    def of(cls, generation: Generation) -> Self:
        """Score the suffixes generated for one prefix against the ground truth they continue.

        Args:
            generation: The model's answer for one prefix, decoded into the log's own units.
        Returns:
            The prefix's scores. A prefix with no samples scores 0.0 on everything, the worst it
            can be, rather than looking like a perfect prediction.
        """
        samples, point, truth = generation.samples, generation.point, generation.truth

        # One similarity per distinct suffix, weighed by how many draws took it: a draw repeated is
        # the same distance from the truth every time, so this is the mean over the draws with the
        # edit distance solved once instead of once per draw.
        similarities = [
            sequence_similarity(suffix, truth.activities) for suffix in samples.suffixes
        ]
        draws = len(samples)

        return cls(
            dls_mean=(
                float(samples.counts @ similarities) / draws if similarities and draws else 0.0
            ),
            dls_point=sequence_similarity(point.activities, truth.activities),
            dls_best=max(similarities, default=0.0),
            hit_rate_at_1=is_hit(samples=samples, truth=truth.activities, k=1),
            hit_rate_at_5=is_hit(samples=samples, truth=truth.activities, k=5),
            hit_rate_at_10=is_hit(samples=samples, truth=truth.activities, k=10),
            remaining_time_ae_mean_days=mean(
                [
                    abs(events.remaining_time_minutes - truth.remaining_time_minutes)
                    for events in samples.events
                ]
            )
            / MINUTES_PER_DAY,
            remaining_time_ae_point_days=abs(
                point.remaining_time_minutes - truth.remaining_time_minutes
            )
            / MINUTES_PER_DAY,
            cycle_time_ae_mean_days=mean(
                [
                    cycle_time_ae_minutes(
                        predicted=events.cycle_time_minutes,
                        true=truth.cycle_time_minutes,
                    )
                    for events in samples.events
                ]
            )
            / MINUTES_PER_DAY,
            cycle_time_ae_point_days=cycle_time_ae_minutes(
                predicted=point.cycle_time_minutes,
                true=truth.cycle_time_minutes,
            )
            / MINUTES_PER_DAY,
            length_ae_mean=mean(
                [float(abs(len(events) - len(truth))) for events in samples.events]
            ),
            length_ae_point=float(abs(len(point) - len(truth))),
            suffix_length=float(len(truth)),
        )


def cycle_time_ae_minutes(predicted: Sequence[float], true: Sequence[float]) -> float:
    """Mean absolute error between a run's cycle times before each event and the true ones, in
    minutes.

    The true suffix sets the range the error is read over: a run that ended early counts as 0
    minutes at every position it did not write, and anything it wrote past the true end is dropped.

    Args:
        predicted: The generated minutes until each event of one run.
        true: The minutes until each event of the true suffix.
    Returns:
        The mean absolute error over the `len(true)` positions, or 0.0 for an empty true suffix.
    """
    if not true:
        return 0.0
    padded = islice(chain(predicted, repeat(0.0)), len(true))
    return mean([abs(prediction - actual) for prediction, actual in zip(padded, true, strict=True)])


def is_hit(samples: Draws, truth: str, *, k: int) -> float:
    """
    Whether the true sequence is among the first `k` samples, exactly.

    Args:
        samples: The suffixes drawn for one prefix. Read through `taken`, which is in the order the
            draws were taken: a draw is independent of the ones before it, so the first `k` of them
            are as good a sample of `k` as any other choice.
        truth: The ground-truth suffix, coded.
        k: How many samples to look at.
    Returns:
        1.0 if one of the first `k` samples is the true sequence, 0.0 otherwise.
    """
    return float(any(samples.suffixes[index] == truth for index in samples.taken[:k]))
