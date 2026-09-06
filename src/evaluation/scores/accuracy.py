from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain, islice, repeat
from typing import Self

import numpy as np

from src.inference.generation import Draws, Generation
from src.scalar_metrics import Direction, Owner, ScalarMetrics, Unit, mean, metric
from src.suffixes import sequence_similarity

MINUTES_PER_DAY = 1440.0

# What the central intervals a coverage is read off cover. Three levels rather than one: a model
# whose spread is the wrong shape rather than the wrong width misses them by different amounts,
# which one level cannot say.
COVERAGE_LEVELS = (0.5, 0.75, 0.95)


@dataclass(frozen=True, slots=True)
class AccuracyScores(ScalarMetrics):
    """How close a prefix's generated suffixes are to the one that actually followed it.

    The scores of one prefix's generated samples, or their mean over a set of prefixes.
    """

    # The Damerau-Levenshtein Similarity (DLS) between the samples and the ground truth.
    # The mean similarity of a prefix's samples to the ground truth. A diagnostic rather than an
    # answer: it is `E[d(draw, truth)]`, the expected accuracy of one draw, which is won by putting
    # every draw on one suffix, so it says nothing about a set of them. `dls-by-length` draws it
    # against the point estimate; no table holds it.
    dls_mean: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    # z = mean(p(z | prefix), a single greedy answer, scored against the ground truth
    dls_point: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    # The closest of a prefix's samples to the ground truth.
    dls_best: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)

    # The share of prefixes whose true suffix is exactly among their first k samples.
    hit_rate_at_1: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    hit_rate_at_5: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    hit_rate_at_10: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    # The same read over every draw rather than the first k, so a mean is the share of prefixes
    # whose true suffix the model wrote at all.
    hit_rate_any: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)

    # What share of a prefix's draws were exactly the true suffix, which is the probability the
    # model's draws put on the continuation the log took. Where `hit_rate_any` asks whether the
    # truth was reachable at all, this asks how much of the model's mass sits on it.
    hit_share: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)

    # The Continuous Ranked Probability Score of the draws against the truth, on the three marginals
    # a suffix carries beyond its activities. `E|X - y| - 0.5 * E|X - X'|`: proper, so it is not won
    # by collapsing, and exactly the absolute error for a single draw, which is what puts each of
    # these on one scale with the `_ae_point` column beside it.
    length_crps: float = metric(unit=Unit.EVENTS, direction=Direction.LOWER)
    remaining_time_crps_days: float = metric(unit=Unit.DAYS, direction=Direction.LOWER)
    cycle_time_crps_days: float = metric(unit=Unit.DAYS, direction=Direction.LOWER)

    # Whether the truth fell inside the central interval of the draws, less the level that interval
    # covers, so a mean over prefixes is the empirical coverage minus the nominal one and 0 is a
    # model whose spread is honest. Negative is over-confident, positive over-dispersed.
    length_coverage_gap_50: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    length_coverage_gap_75: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    length_coverage_gap_95: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    remaining_time_coverage_gap_50: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    remaining_time_coverage_gap_75: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    remaining_time_coverage_gap_95: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    cycle_time_coverage_gap_50: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    cycle_time_coverage_gap_75: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)
    cycle_time_coverage_gap_95: float = metric(unit=Unit.SCORE, direction=Direction.ZERO)

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

        drawn_truth = truth.activities in samples.suffixes
        share_of_truth = (
            float(samples.counts[samples.suffixes.index(truth.activities)]) / draws
            if drawn_truth and draws
            else 0.0
        )

        # The three marginals a suffix carries beyond its activities, one row per draw and one
        # column per value that marginal holds: `[draws, 1]` for the two a suffix carries once and
        # `[draws, len(truth)]` for the cycle times, so `crps` and `coverage_gap` read all three the
        # same way. Read per draw rather than per distinct suffix: two draws of one suffix came from
        # different `z` and carry their own times.
        lengths = np.array(
            [[float(len(events))] for events in samples.events], dtype=np.float64
        ).reshape(draws, 1)
        remaining = np.array(
            [[events.remaining_time_minutes] for events in samples.events], dtype=np.float64
        ).reshape(draws, 1)
        # A draw that ended early is padded with 0.0 and one that ran on is cut at the true end,
        # the alignment `cycle_time_ae_minutes` reads on.
        cycle_times = np.array(
            [
                aligned_cycle_times(events.cycle_time_minutes, length=len(truth))
                for events in samples.events
            ],
            dtype=np.float64,
        ).reshape(draws, len(truth))

        # [1], [1] and [len(truth)], matching the columns of the three above.
        true_length = np.array([float(len(truth))], dtype=np.float64)
        true_remaining = np.array([truth.remaining_time_minutes], dtype=np.float64)
        true_cycle_times = np.array(truth.cycle_time_minutes, dtype=np.float64)

        length_gaps = _coverage_gaps(lengths, true_length)
        remaining_gaps = _coverage_gaps(remaining, true_remaining)
        cycle_gaps = _coverage_gaps(cycle_times, true_cycle_times)

        return cls(
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
    """One run's cycle times cut to the positions the true suffix covers.

    The one place that alignment is decided, read by every cycle-time score: a run that ended early
    counts as 0 minutes at every position it did not write, and anything it wrote past the true end
    is dropped.

    Args:
        predicted: The generated minutes until each event of one run.
        length: How many events the true suffix holds.
    Returns:
        Exactly `length` values.
    """
    return list(islice(chain(predicted, repeat(0.0)), length))


def cycle_time_ae_minutes(predicted: Sequence[float], true: Sequence[float]) -> float:
    """Mean absolute error between a run's cycle times before each event and the true ones, in
    minutes.

    Args:
        predicted: The generated minutes until each event of one run.
        true: The minutes until each event of the true suffix.
    Returns:
        The mean absolute error over the `len(true)` positions, or 0.0 for an empty true suffix.
    """
    if not true:
        return 0.0
    padded = aligned_cycle_times(predicted, length=len(true))
    return mean([abs(prediction - actual) for prediction, actual in zip(padded, true, strict=True)])


def crps(draws: np.ndarray, truth: np.ndarray) -> float:
    """The Continuous Ranked Probability Score of a set of draws against the values observed.

    `E|X - y| - 0.5 * E|X - X'|`, the draws read as the predictive distribution they are. Proper,
    so it is not won by collapsing: the first term alone is the expected accuracy of one draw and is
    best when every draw sits on the mode, which the second term is what charges for. For a single
    draw it is exactly `|x - y|`, which is what puts a CRPS column and the absolute error of a point
    prediction on one scale, where `dls_mean` and `dls_point` are two questions and never share one.

    The spread term is the mean over the ordered pairs of two *distinct* draws, the convention
    `src.suffixes.diversity` reads on, so a smaller ensemble is not rewarded for being one.

    Read column by column so that a marginal a suffix carries once and one it carries per position
    are the same call: the first is a single column, the second one per position of the true suffix.

    Args:
        draws: `[draws, columns]`, the values drawn for one prefix.
        truth: `[columns]`, the value the log actually took in each.
    Returns:
        The mean score over the columns, in the unit of `draws`: 0.0 where every draw is the truth
        and rising from there. 0.0 for a prefix with no draws, as every other score of one reads,
        and for a true suffix with no events, as `cycle_time_ae_minutes` reads there.
    """
    count, columns = draws.shape
    if count == 0 or columns == 0:
        return 0.0
    # [columns]
    accuracy = np.abs(draws - truth).mean(axis=0)
    if count == 1:
        return float(accuracy.mean())

    # `sum over i < j of (x_(j) - x_(i))` per column, which is what that column's full pairwise
    # matrix sums to, in one pass over its sorted values.
    ordered = np.sort(draws, axis=0)
    # [draws, 1], so the weights broadcast down every column at once.
    ranks = np.arange(count, dtype=np.float64)[:, None]
    spread = ((2.0 * ranks - count + 1.0) * ordered).sum(axis=0)
    return float((accuracy - spread / (count * (count - 1.0))).mean())


def coverage_gap(draws: np.ndarray, truth: np.ndarray, *, level: float) -> float:
    """Whether the truth fell inside the central `level` interval of the draws, less `level`.

    Read as a gap rather than as a share because the value a coverage should have is the level it
    was taken at rather than 1.0: a model whose 95% interval catches everything is as wrong as one
    whose interval catches nothing, only in the other direction. A mean over prefixes is the
    empirical coverage minus the nominal one, best at 0 and signed, so it says which way a model
    is wrong: negative is over-confident, positive over-dispersed.

    The interval is the two empirical quantiles of the draws. The draws are the only spread that
    reaches evaluation, the time heads' scale not being written to the generations file, so there
    is no parametric interval to read instead.

    That leaves the gap biased low, an interval read off a finite sample being narrower than the one
    it estimates. A calibrated model drawing 100 reads about -0.012 at 50, -0.016 at 75 and -0.018
    at 95, and the bias vanishes with the draw count: at 400 it is under -0.005, at 2000 under
    -0.001. It follows the draw count alone, which every model of a log shares, so this is a number
    to compare models on rather than a miscalibration to quote on its own, the way
    `activity_time_wasserstein_days` is.

    Read column by column, as `crps` is, so one call covers a marginal a suffix carries once and one
    it carries per position alike.

    Args:
        draws: `[draws, columns]`, the values drawn for one prefix.
        truth: `[columns]`, the value the log actually took in each.
        level: What the interval covers, in `(0, 1)`.
    Returns:
        The share of the columns whose truth fell inside, less `level`. `-level` for a prefix with
        no draws, the worst it can read, and 0.0 for a true suffix with no events, there being no
        interval to have missed.
    """
    count, columns = draws.shape
    if columns == 0:
        return 0.0
    if count == 0:
        return -level
    # [columns] each
    low, high = np.quantile(draws, [(1.0 - level) / 2.0, (1.0 + level) / 2.0], axis=0)
    return float(((low <= truth) & (truth <= high)).mean()) - level


def _coverage_gaps(draws: np.ndarray, truth: np.ndarray) -> list[float]:
    """One coverage gap per level of `COVERAGE_LEVELS`, over the columns of one marginal."""
    return [coverage_gap(draws, truth, level=level) for level in COVERAGE_LEVELS]


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
