from dataclasses import dataclass
from typing import Self, Sequence

import pandas as pd
from Declare4Py.ProcessModels.DeclareModel import DeclareModel

from evaluation.accuracy import AccuracyScores, score_generation
from src.evaluation.conformance import ConformanceScores, score_conformance
from src.inference import generation_from_rows


@dataclass(frozen=True)
class PrefixScores:
    """One prefix's scores, or their mean over a set of prefixes.

    Accuracy asks how close a generated suffix is to the one that actually happened; conformance
    asks whether it is a trace the process allows at all. A suffix can score well on one and badly
    on the other, which is why they are carried side by side rather than folded together.
    """
    accuracy: AccuracyScores
    conformance: ConformanceScores

    @classmethod
    def mean(cls, values: Sequence[Self]) -> Self:
        """Average a set of prefix scores, each family averaged by its own rules.

        Args:
            values: The scores to average, one per prefix.
        Returns:
            The mean accuracy beside the mean conformance.
        """
        return cls(
            accuracy=AccuracyScores.mean([value.accuracy for value in values]),
            conformance=ConformanceScores.mean([value.conformance for value in values]),
        )


@dataclass(frozen=True)
class ByPrefixLengthMetrics:
    """The scores of the prefixes of one length, and how many pairs that length had."""
    length: int
    pairs_count: int
    scores: PrefixScores


@dataclass(frozen=True)
class EvaluationMetrics:
    """The scores over every prefix, and the same scores broken down by prefix length.

    The breakdown is worth keeping: a model that only works once most of the case is already known
    looks the same as a good one in the headline average, and the errors that scale with how much
    case is left cannot be read at all from a pooled number. Conformance follows the same cut,
    where the shorter the prefix the more the generated suffix has to earn the rate by itself.
    """
    scores: PrefixScores
    # In increasing order of prefix length
    by_prefix_length: list[ByPrefixLengthMetrics]


def evaluate_generations(
    generations: pd.DataFrame,
    *,
    declare_model: DeclareModel,
    consider_vacuity: bool,
) -> EvaluationMetrics:
    """
    Score generated suffixes against the ground truth they were generated for, and against the
    declarative model the dataset was mined for.

    Args:
        generations: Rows written by `src/inference/writer.py`, with the truncated pairs
            already dropped (their ground-truth suffix stops short of the real ending, so
            nothing here would be measuring what it claims to, and a constraint checker would
            read them as finished cases they are not).
        declare_model: The model to check conformance against, from `load_declare_model`.
        consider_vacuity: Whether a constraint a trace never activates counts as satisfied.
    Returns:
        The averages over every prefix, and the same averages broken down by cut point. Every
        prefix weighs the same however many samples were drawn for it, so a prefix is the unit
        the report describes and a row is not.
    """
    # Each group is scored down to a handful of floats and dropped, so a split of millions of rows
    # never has more than one prefix's objects alive at a time. Bucketing by cut point as they are
    # scored is what saves a second pass over them afterwards.
    buckets: dict[int, list[PrefixScores]] = {}
    for (_, prefix_len), group in generations.groupby(['case_id', 'prefix_len'], sort=False):
        generation = generation_from_rows(group)
        buckets.setdefault(int(prefix_len), []).append(
            PrefixScores(
                accuracy=score_generation(generation),
                conformance=score_conformance(
                    generation, model=declare_model, consider_vacuity=consider_vacuity
                ),
            )
        )

    return EvaluationMetrics(
        scores=PrefixScores.mean([scores for bucket in buckets.values() for scores in bucket]),
        by_prefix_length=[
            ByPrefixLengthMetrics(
                length=length,
                pairs_count=len(buckets[length]),
                scores=PrefixScores.mean(buckets[length]),
            )
            for length in sorted(buckets)
        ],
    )
