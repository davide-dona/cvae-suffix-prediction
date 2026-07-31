from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from src.evaluation.sequences import sequence_similarity


@dataclass(frozen=True)
class PrefixLengthAccuracy:
    """How well the suffixes of one cut point were predicted."""
    pairs: int
    activity_dls_mean: float


@dataclass(frozen=True)
class AccuracyMetrics:
    """How close the generated suffixes came to the ground truth.

    Every field is an average over prefixes of a per-prefix number, which is itself either the
    mean or the best over that prefix's samples. The two are worth reading together: `_mean` is
    what one draw is worth, `_best` is whether the model covers the truth at all, and the gap
    between them is what the latent is contributing.
    """
    activity_dls_mean: float
    activity_dls_best: float
    remaining_time_ae_mean_minutes: float
    remaining_time_ae_best_minutes: float
    length_ae_mean: float
    length_ae_best: float
    by_prefix_length: dict[int, PrefixLengthAccuracy]


@dataclass(frozen=True)
class _PrefixAccuracy:
    """The same six numbers, for the samples of a single prefix."""
    prefix_len: int
    activity_dls_mean: float
    activity_dls_best: float
    remaining_time_ae_mean_minutes: float
    remaining_time_ae_best_minutes: float
    length_ae_mean: float
    length_ae_best: float


def accuracy_metrics(predictions: pd.DataFrame) -> AccuracyMetrics:
    """
    Score generated suffixes against the ground truth they were generated for.

    Prefixes are weighted equally, whatever the case they were cut from: the mean is taken over
    prefixes, not over rows, so a prefix does not count more for having more samples.

    Args:
        predictions: Rows written by `src/inference/predict.py`, with the truncated pairs
            already dropped (their ground-truth suffix stops short of the real ending, so
            nothing here would be measuring what it claims to).
    Returns:
        The averages over every prefix, and the activity similarity broken down by cut point.
    """
    per_prefix = [
        _prefix_accuracy(samples)
        for _, samples in predictions.groupby(['case_id', 'prefix_len'], sort=False)
    ]
    return AccuracyMetrics(
        activity_dls_mean=_mean([p.activity_dls_mean for p in per_prefix]),
        activity_dls_best=_mean([p.activity_dls_best for p in per_prefix]),
        remaining_time_ae_mean_minutes=_mean([p.remaining_time_ae_mean_minutes for p in per_prefix]),
        remaining_time_ae_best_minutes=_mean([p.remaining_time_ae_best_minutes for p in per_prefix]),
        length_ae_mean=_mean([p.length_ae_mean for p in per_prefix]),
        length_ae_best=_mean([p.length_ae_best for p in per_prefix]),
        by_prefix_length=_by_prefix_length(per_prefix),
    )


def _prefix_accuracy(samples: pd.DataFrame) -> _PrefixAccuracy:
    """Reduce the samples generated for one prefix to a mean and a best of each metric.

    Args:
        samples: The rows of a single (case, cut point), one per generated sample. They share
            a ground truth, so it is read off the first of them.
    Returns:
        That prefix's contribution to the averages.
    """
    truth = samples.iloc[0]
    # Suffix deltas are gaps between consecutive events, so their sum is the time left from the
    # end of the prefix to the end of the case: the remaining cycle time.
    true_remaining = float(sum(truth.true_time_deltas_minutes))

    activity_dls: list[float] = []
    remaining_time_ae: list[float] = []
    length_ae: list[float] = []
    for sample in samples.itertuples():
        activity_dls.append(sequence_similarity(sample.predicted_activities, truth.true_activities))
        remaining_time_ae.append(
            abs(float(sum(sample.predicted_time_deltas_minutes)) - true_remaining)
        )
        length_ae.append(float(abs(len(sample.predicted_activities) - len(truth.true_activities))))

    return _PrefixAccuracy(
        prefix_len=int(truth.prefix_len),
        activity_dls_mean=_mean(activity_dls),
        activity_dls_best=max(activity_dls),
        remaining_time_ae_mean_minutes=_mean(remaining_time_ae),
        remaining_time_ae_best_minutes=min(remaining_time_ae),
        length_ae_mean=_mean(length_ae),
        length_ae_best=min(length_ae),
    )


def _by_prefix_length(per_prefix: Sequence[_PrefixAccuracy]) -> dict[int, PrefixLengthAccuracy]:
    """Group the per-prefix scores by cut point, in increasing order.

    Accuracy against prefix length is the breakdown worth keeping: a model that only works once
    most of the case is already known looks the same as a good one in the headline average.
    """
    scores: dict[int, list[float]] = {}
    for prefix in per_prefix:
        scores.setdefault(prefix.prefix_len, []).append(prefix.activity_dls_mean)

    return {
        prefix_len: PrefixLengthAccuracy(
            pairs=len(scores[prefix_len]), activity_dls_mean=_mean(scores[prefix_len])
        )
        for prefix_len in sorted(scores)
    }


def _mean(values: Sequence[float]) -> float:
    """The mean of `values`, or 0.0 if there are none."""
    return sum(values) / len(values) if values else 0.0
