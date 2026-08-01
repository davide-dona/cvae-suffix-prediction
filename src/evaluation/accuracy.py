from dataclasses import dataclass
from typing import Sequence
import pandas as pd

from src.evaluation.sequences import mean, score_samples


@dataclass(frozen=True)
class PrefixLengthAccuracy:
    """How well the suffixes of one cut point were predicted."""
    pairs: int
    activity_dls_mean: float


@dataclass(frozen=True)
class AccuracyMetrics:
    """How close the generated suffixes came to the ground truth.

    Every field is an average over prefixes of a per-prefix number, which is itself either the
    mean or the best over that prefix's samples. The two are worth reading together: `mean` is
    what one draw is worth, `best` is whether the model covers the truth at all, and the gap
    between them is what the latent is contributing.

    `sample_diversity` stands apart: it is not a mean/best pair, since it needs every sample of a
    prefix at once rather than reducing over them independently.
    """
    activity_dls_mean: float
    activity_dls_best: float
    remaining_time_ae_mean_minutes: float
    remaining_time_ae_best_minutes: float
    length_ae_mean: float
    length_ae_best: float
    sample_diversity: float
    by_prefix_length: dict[int, PrefixLengthAccuracy]


@dataclass(frozen=True)
class _PrefixAccuracy:
    """The same numbers, for the samples of a single prefix."""
    prefix_len: int
    activity_dls_mean: float
    activity_dls_best: float
    remaining_time_ae_mean_minutes: float
    remaining_time_ae_best_minutes: float
    length_ae_mean: float
    length_ae_best: float
    sample_diversity: float


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
        activity_dls_mean=mean([p.activity_dls_mean for p in per_prefix]),
        activity_dls_best=mean([p.activity_dls_best for p in per_prefix]),
        remaining_time_ae_mean_minutes=mean([p.remaining_time_ae_mean_minutes for p in per_prefix]),
        remaining_time_ae_best_minutes=mean([p.remaining_time_ae_best_minutes for p in per_prefix]),
        length_ae_mean=mean([p.length_ae_mean for p in per_prefix]),
        length_ae_best=mean([p.length_ae_best for p in per_prefix]),
        sample_diversity=mean([p.sample_diversity for p in per_prefix]),
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
    # The remaining cycle time: minutes from the end of the prefix to the end of the case,
    # which the model predicts directly rather than as a sum of per-event gaps.
    true_remaining = float(truth.true_remaining_time_minutes)

    remaining_time_ae: list[float] = []
    length_ae: list[float] = []
    predicted_activities: list[list[str]] = []
    for sample in samples.itertuples():
        remaining_time_ae.append(
            abs(float(sample.predicted_remaining_time_minutes) - true_remaining)
        )
        length_ae.append(float(abs(len(sample.predicted_activities) - len(truth.true_activities))))
        predicted_activities.append(sample.predicted_activities)

    activities = score_samples(predicted_activities, truth.true_activities)
    return _PrefixAccuracy(
        prefix_len=int(truth.prefix_len),
        activity_dls_mean=activities.dls_mean,
        activity_dls_best=activities.dls_best,
        remaining_time_ae_mean_minutes=mean(remaining_time_ae),
        remaining_time_ae_best_minutes=min(remaining_time_ae),
        length_ae_mean=mean(length_ae),
        length_ae_best=min(length_ae),
        sample_diversity=activities.sample_diversity,
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
            pairs=len(scores[prefix_len]), activity_dls_mean=mean(scores[prefix_len])
        )
        for prefix_len in sorted(scores)
    }
