from dataclasses import dataclass
from typing import Sequence
import pandas as pd

from src.inference import generation_from_rows
from src.scoring import SuffixScores, score_prefix


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
        generations: Rows written by `src/inference/writer.py`, with the truncated pairs
            already dropped (their ground-truth suffix stops short of the real ending, so
            nothing here would be measuring what it claims to).
    Returns:
        The averages over every prefix, and the same averages broken down by cut point. Every
        prefix weighs the same however many samples were drawn for it, so a prefix is the unit
        the report describes and a row is not.
    """
    # Each group is scored down to eleven floats and dropped, so a split of millions of rows never
    # has more than one prefix's objects alive at a time.
    per_prefix = [
        _PrefixAccuracy(
            prefix_len=int(prefix_len), scores=score_prefix(generation_from_rows(group))
        )
        for (_, prefix_len), group in generations.groupby(['case_id', 'prefix_len'], sort=False)
    ]
    return AccuracyMetrics(
        scores=SuffixScores.mean([prefix.scores for prefix in per_prefix]),
        by_prefix_length=_by_prefix_length(per_prefix),
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
            scores=SuffixScores.mean(buckets[prefix_len]),
        )
        for prefix_len in sorted(buckets)
    ]
