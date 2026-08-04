from dataclasses import dataclass, fields
from typing import Sequence
import pandas as pd

from src.evaluation.sequences import mean, score_samples


@dataclass(frozen=True)
class SuffixScores:
    """The scores of one prefix's generated samples, or their mean over a set of prefixes."""
    # The Damerau-Levenshtein Similarity (DLS) is the edit distance normalized to [0, 1] and
    # inverted, so 1.0 is perfect. `mean` is what one draw is worth, `best` whether the model
    # covers the truth at all.
    activity_dls_mean: float
    activity_dls_best: float

    # The samples read as a predictive distribution over suffixes, lower being better. Unlike
    # `sample_diversity` and `unique_sample_rate` below, which have no target value, this one can
    # be compared across runs and minimized.
    activity_energy_score: float

    # How far apart the samples of a prefix are, and how many of them are distinct sequences.
    # Both say how much of the prefix's uncertainty z carries, not how good the model is.
    sample_diversity: float
    unique_sample_rate: float

    # Absolute error (AE) between the predicted and true remaining cycle time, in minutes
    remaining_time_ae_mean_minutes: float
    remaining_time_ae_best_minutes: float

    # AE between the predicted and true suffix length, in events
    length_ae_mean: float
    length_ae_best: float

    # Events left after the cut point: the scale every error above is read against
    suffix_length: float


@dataclass(frozen=True)
class ByPrefixLengthAccuracy:
    """The scores of the prefixes of one length, and how many pairs that length had."""
    length: int
    pairs_count: int
    scores: SuffixScores


@dataclass(frozen=True)
class AccuracyMetrics:
    """The scores over every prefix, and the same scores broken down by prefix length."""
    scores: SuffixScores
    # In increasing order of prefix length
    by_prefix_length: list[ByPrefixLengthAccuracy]


@dataclass(frozen=True)
class _PrefixAccuracy:
    """One prefix's scores, kept beside the cut point they were measured at."""
    prefix_len: int
    scores: SuffixScores


def accuracy_metrics(generations: pd.DataFrame) -> AccuracyMetrics:
    """
    Score generated suffixes against the ground truth they were generated for.
    Args:
        generations: Rows written by `src/inference/generate.py`, with the truncated pairs
            already dropped (their ground-truth suffix stops short of the real ending, so
            nothing here would be measuring what it claims to).
    Returns:
        The averages over every prefix, and the same averages broken down by cut point. Every
        prefix weighs the same however many samples were drawn for it, so a prefix is the unit
        the report describes and a row is not.
    """
    per_prefix = [
        _prefix_accuracy(samples)
        for _, samples in generations.groupby(['case_id', 'prefix_len'], sort=False)
    ]
    return AccuracyMetrics(
        scores=mean_scores([prefix.scores for prefix in per_prefix]),
        by_prefix_length=_by_prefix_length(per_prefix),
    )


def mean_scores(scores: Sequence[SuffixScores]) -> SuffixScores:
    """Average a set of prefixes' scores field by field.

    Args:
        scores: The scores to average, one per prefix.
    Returns:
        The mean of every field, all 0.0 if there are no scores.
    """
    return SuffixScores(**{
        field.name: mean([getattr(score, field.name) for score in scores])
        for field in fields(SuffixScores)
    })


def _prefix_accuracy(samples: pd.DataFrame) -> _PrefixAccuracy:
    """Reduce the S samples of a single cut point to a single score, by averaging the per-sample scores.
    Args:
        samples: The rows of a single (case, cut point), one per generated sample. They share
            a ground truth, so it is read off the first of them.
    Returns:
        That prefix's contribution to the averages.
    """
    ground_truth = samples.iloc[0]

    remaining_time_ae: list[float] = []
    length_ae: list[float] = []
    generated_activities: list[list[str]] = []

    for sample in samples.itertuples():
        remaining_time_ae.append(
            abs(float(sample.generated_remaining_time_minutes) - float(ground_truth.true_remaining_time_minutes))
        )
        length_ae.append(float(abs(len(sample.generated_activities) - len(ground_truth.true_activities))))
        generated_activities.append(sample.generated_activities)

    activities = score_samples(generated_activities, ground_truth.true_activities)
    return _PrefixAccuracy(
        prefix_len=int(ground_truth.prefix_len),
        scores=SuffixScores(
            activity_dls_mean=activities.dls_mean,
            activity_dls_best=activities.dls_best,
            activity_energy_score=activities.energy_score,
            sample_diversity=activities.sample_diversity,
            unique_sample_rate=activities.unique_sample_rate,
            remaining_time_ae_mean_minutes=mean(remaining_time_ae),
            remaining_time_ae_best_minutes=min(remaining_time_ae),
            length_ae_mean=mean(length_ae),
            length_ae_best=min(length_ae),
            suffix_length=float(len(ground_truth.true_activities)),
        ),
    )


def _by_prefix_length(per_prefix: Sequence[_PrefixAccuracy]) -> list[ByPrefixLengthAccuracy]:
    """Group the per-prefix scores by cut point, in increasing order.

    Accuracy against prefix length is the breakdown worth keeping: a model that only works once
    most of the case is already known looks the same as a good one in the headline average, and
    the errors that scale with how much case is left cannot be read at all from a pooled number.
    """
    buckets: dict[int, list[SuffixScores]] = {}
    for prefix in per_prefix:
        buckets.setdefault(prefix.prefix_len, []).append(prefix.scores)

    return [
        ByPrefixLengthAccuracy(
            length=prefix_len,
            pairs_count=len(buckets[prefix_len]),
            scores=mean_scores(buckets[prefix_len]),
        )
        for prefix_len in sorted(buckets)
    ]
