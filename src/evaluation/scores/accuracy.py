from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain, islice, repeat
from typing import Self

import numpy as np

from src.inference.generation import Draws, Generation
from src.scalar_metrics import Direction, Owner, ScalarMetrics, Unit, mean, metric
from src.suffixes import distances, diversity, sequence_similarity

MINUTES_PER_DAY = 1440.0

# Central interval levels used for calibration.
COVERAGE_LEVELS = (0.5, 0.75, 0.95)


@dataclass(frozen=True, slots=True)
class AccuracyScores(ScalarMetrics):
    """Accuracy of generated suffixes against the observed continuation."""

    energy_score: float = metric(unit=Unit.SCORE, direction=Direction.LOWER)

    # Mean Damerau-Levenshtein similarity across draws.
    dls_mean: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    # Point prediction from the latent mean.
    dls_point: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    # Best sampled similarity to the ground truth.
    dls_best: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)

    # Exact-match rate among the first k draws.
    hit_rate_at_1: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    hit_rate_at_5: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    hit_rate_at_10: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    # Whether any draw exactly matches the truth.
    hit_rate_any: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)

    # Fraction of draws that exactly match the truth.
    hit_share: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)

    # CRPS on suffix length, remaining time, and per-event cycle time.
    length_crps: float = metric(unit=Unit.EVENTS, direction=Direction.LOWER)
    remaining_time_crps_days: float = metric(unit=Unit.DAYS, direction=Direction.LOWER)
    cycle_time_crps_days: float = metric(unit=Unit.DAYS, direction=Direction.LOWER)

    # Empirical minus nominal central-interval coverage.
    length_coverage_gap_50: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    length_coverage_gap_75: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    length_coverage_gap_95: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    remaining_time_coverage_gap_50: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    remaining_time_coverage_gap_75: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    remaining_time_coverage_gap_95: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    cycle_time_coverage_gap_50: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    cycle_time_coverage_gap_75: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    cycle_time_coverage_gap_95: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)

    # Remaining-time absolute error in days.
    remaining_time_ae_mean_days: float = metric(unit=Unit.DAYS, direction=Direction.LOWER)
    remaining_time_ae_point_days: float = metric(unit=Unit.DAYS, direction=Direction.LOWER)

    # Per-event cycle-time absolute error in days.
    cycle_time_ae_mean_days: float = metric(unit=Unit.DAYS, direction=Direction.LOWER)
    cycle_time_ae_point_days: float = metric(unit=Unit.DAYS, direction=Direction.LOWER)

    # Suffix-length absolute error in events.
    length_ae_mean: float = metric(unit=Unit.EVENTS, direction=Direction.LOWER)
    length_ae_point: float = metric(unit=Unit.EVENTS, direction=Direction.LOWER)

    # Ground-truth suffix length, owned by the log.
    suffix_length: float = metric(unit=Unit.EVENTS, owner=Owner.LOG)

    @classmethod
    def of(cls, generation: Generation) -> Self:
        """Score one generation against its ground-truth suffix.

        Args:
            generation: Decoded model output for one prefix.

        Returns:
            Accuracy scores for the prefix.
        """
        samples, point, truth = generation.samples, generation.point, generation.truth

        # Score distinct suffixes once, then weight by draw count.
        similarities = [
            sequence_similarity(suffix, truth.activities) for suffix in samples.suffixes
        ]
        draws = len(samples)

        drawn_truth = truth.activities in samples.suffixes
        share_of_truth = (
            float(samples.counts[samples.suffixes.index(truth.activities)]) / draws
            if drawn_truth and draws
            else 0.0
        )

        # Represent scalar and per-event marginals as draw-by-column arrays.
        lengths = np.array(
            [[float(len(events))] for events in samples.events], dtype=np.float64
        ).reshape(draws, 1)
        remaining = np.array(
            [[events.remaining_time_minutes] for events in samples.events], dtype=np.float64
        ).reshape(draws, 1)
        # Align generated cycle times to the ground-truth length.
        cycle_times = np.array(
            [
                aligned_cycle_times(events.cycle_time_minutes, length=len(truth))
                for events in samples.events
            ],
            dtype=np.float64,
        ).reshape(draws, len(truth))

        # One truth value per marginal column.
        true_length = np.array([float(len(truth))], dtype=np.float64)
        true_remaining = np.array([truth.remaining_time_minutes], dtype=np.float64)
        true_cycle_times = np.array(truth.cycle_time_minutes, dtype=np.float64)

        length_gaps = _coverage_gaps(lengths, true_length)
        remaining_gaps = _coverage_gaps(remaining, true_remaining)
        cycle_gaps = _coverage_gaps(cycle_times, true_cycle_times)

        return cls(
            energy_score=energy_score(samples, truth.activities),
            dls_mean=(
                float(samples.counts @ similarities) / draws if similarities and draws else 0.0
            ),
            dls_point=sequence_similarity(point.activities, truth.activities),
            dls_best=max(similarities, default=0.0),
            hit_rate_at_1=is_hit(samples=samples, truth=truth.activities, k=1),
            hit_rate_at_5=is_hit(samples=samples, truth=truth.activities, k=5),
            hit_rate_at_10=is_hit(samples=samples, truth=truth.activities, k=10),
            hit_rate_any=float(drawn_truth),
            hit_share=share_of_truth,
            length_crps=crps(lengths, true_length),
            remaining_time_crps_days=crps(remaining, true_remaining) / MINUTES_PER_DAY,
            cycle_time_crps_days=crps(cycle_times, true_cycle_times) / MINUTES_PER_DAY,
            length_coverage_gap_50=length_gaps[0],
            length_coverage_gap_75=length_gaps[1],
            length_coverage_gap_95=length_gaps[2],
            remaining_time_coverage_gap_50=remaining_gaps[0],
            remaining_time_coverage_gap_75=remaining_gaps[1],
            remaining_time_coverage_gap_95=remaining_gaps[2],
            cycle_time_coverage_gap_50=cycle_gaps[0],
            cycle_time_coverage_gap_75=cycle_gaps[1],
            cycle_time_coverage_gap_95=cycle_gaps[2],
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


def aligned_cycle_times(predicted: Sequence[float], *, length: int) -> list[float]:
    """Pad or truncate cycle times to the ground-truth length.

    Args:
        predicted: Generated cycle times in minutes.
        length: Ground-truth suffix length.

    Returns:
        Exactly `length` cycle times.
    """
    return list(islice(chain(predicted, repeat(0.0)), length))


def cycle_time_ae_minutes(predicted: Sequence[float], true: Sequence[float]) -> float:
    """Return cycle-time mean absolute error in minutes.

    Args:
        predicted: Generated cycle times.
        true: Ground-truth cycle times.

    Returns:
        Mean absolute error over ground-truth positions.
    """
    if not true:
        return 0.0
    padded = aligned_cycle_times(predicted, length=len(true))
    return mean([abs(prediction - actual) for prediction, actual in zip(padded, true, strict=True)])


def crps(draws: np.ndarray, truth: np.ndarray) -> float:
    """Return mean CRPS over columns of a draw-by-column array.

    Args:
        draws: Values with shape `[draws, columns]`.
        truth: Observed values with shape `[columns]`.

    Returns:
        Mean continuous ranked probability score.
    """
    count, columns = draws.shape
    if count == 0 or columns == 0:
        return 0.0
    # One accuracy value per column.
    accuracy = np.abs(draws - truth).mean(axis=0)
    if count == 1:
        return float(accuracy.mean())

    # Compute pairwise spread from sorted values without a full matrix.
    ordered = np.sort(draws, axis=0)
    # Broadcast ranks across columns.
    ranks = np.arange(count, dtype=np.float64)[:, None]
    spread = ((2.0 * ranks - count + 1.0) * ordered).sum(axis=0)
    return float((accuracy - spread / (count * (count - 1.0))).mean())


def coverage_gap(draws: np.ndarray, truth: np.ndarray, *, level: float) -> float:
    """Return empirical minus nominal central-interval coverage.

    Args:
        draws: Values with shape `[draws, columns]`.
        truth: Observed values with shape `[columns]`.
        level: Nominal central-interval coverage.

    Returns:
        Empirical coverage minus `level`.
    """
    count, columns = draws.shape
    if columns == 0:
        return 0.0
    if count == 0:
        return -level
    # Lower and upper bounds per column.
    low, high = np.quantile(draws, [(1.0 - level) / 2.0, (1.0 + level) / 2.0], axis=0)
    return float(((low <= truth) & (truth <= high)).mean()) - level


def _coverage_gaps(draws: np.ndarray, truth: np.ndarray) -> list[float]:
    """Return coverage gaps at the configured levels.

    Args:
        draws: Values with shape `[draws, columns]`.
        truth: Observed values with shape `[columns]`.

    Returns:
        One coverage gap per configured level.
    """
    return [coverage_gap(draws, truth, level=level) for level in COVERAGE_LEVELS]


def is_hit(samples: Draws, truth: str, *, k: int) -> float:
    """Return whether the truth appears in the first `k` draws.

    Args:
        samples: Generated suffix draws in draw order.
        truth: Encoded ground-truth suffix.
        k: Number of draws to inspect.

    Returns:
        1.0 if the truth occurs, otherwise 0.0.
    """
    return float(any(samples.suffixes[index] == truth for index in samples.taken[:k]))


def energy_score(samples: Draws, truth: str) -> float:
    """Unbiased sequence energy estimate using normalized OSA distance.

    The spread term averages distinct draw indices, including repeated sequences
    with their multiplicities. This estimate can be negative. Strict propriety
    is not established for normalized OSA distance.
    """
    if len(samples) < 2:
        raise ValueError('energy_score requires at least two draws')
    distances_to_truth = distances(queries=samples.suffixes, choices=[truth], dtype=np.float64)[
        :, 0
    ]
    accuracy = float(samples.counts @ distances_to_truth) / len(samples)
    return accuracy - 0.5 * diversity(samples.suffixes, weights=samples.counts)
