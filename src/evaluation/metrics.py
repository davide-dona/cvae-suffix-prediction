from dataclasses import dataclass
from typing import Iterable, Self, Sequence

from Declare4Py.ProcessModels.DeclareModel import DeclareModel

from src.evaluation.accuracy import AccuracyScores, score_generation
from src.evaluation.conformance import ConformanceScores, score_conformance
from src.inference import Generation


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
    """The scores over every prefix, the same scores broken down by prefix length, and what both
    were measured over.

    The breakdown is worth keeping: a model that only works once most of the case is already known
    looks the same as a good one in the headline average, and the errors that scale with how much
    case is left cannot be read at all from a pooled number. Conformance follows the same cut,
    where the shorter the prefix the more the generated suffix has to earn the rate by itself.

    The counts belong here rather than beside the scores: an average is not comparable across runs
    without knowing how many prefixes went into it and how many were left out, and only the pass
    that scores a file sees them.
    """
    pairs: int
    cases: int
    samples_per_prefix: int
    truncated_pairs_excluded: int

    scores: PrefixScores
    # In increasing order of prefix length
    by_prefix_length: list[ByPrefixLengthMetrics]


def evaluate_generations(
    generations: Iterable[Generation],
    *,
    declare_model: DeclareModel,
    consider_vacuity: bool,
) -> EvaluationMetrics:
    """
    Score generated suffixes against the ground truth they were generated for, and against the
    declarative model the dataset was mined for.

    Truncated prefixes are dropped here rather than scored: their ground-truth suffix stops short
    of the real ending, so nothing measured against it would mean what it claims to, and a
    constraint checker would read them as finished cases they are not.

    Args:
        generations: The prefixes to score, from `read_generations`. Read once, in one pass, so a
            file larger than memory can be scored.
        declare_model: The model to check conformance against, from `load_declare_model`.
        consider_vacuity: Whether a constraint a trace never activates counts as satisfied.
    Returns:
        The averages over every prefix, the same averages broken down by cut point, and the
        population they were taken over. Every prefix weighs the same however many samples were
        drawn for it, so a prefix is the unit this describes and a sample is not.
    """
    # Each prefix is scored down to a handful of floats and dropped, so a split of hundreds of
    # thousands of them never has more than one prefix's objects alive at a time. Bucketing by cut
    # point as they are scored is what saves a second pass over them afterwards.
    buckets: dict[int, list[PrefixScores]] = {}
    cases: set[str] = set()
    truncated_pairs_excluded = 0
    samples_per_prefix = 0

    for generation in generations:
        if generation.truncated:
            truncated_pairs_excluded += 1
            continue

        cases.add(generation.case_id)
        # Every prefix is drawn for the same number of times, so the first answers for all of them.
        samples_per_prefix = samples_per_prefix or len(generation.samples)
        buckets.setdefault(generation.prefix_len, []).append(
            PrefixScores(
                accuracy=score_generation(generation),
                conformance=score_conformance(
                    generation, model=declare_model, consider_vacuity=consider_vacuity
                ),
            )
        )

    scored = [scores for bucket in buckets.values() for scores in bucket]
    return EvaluationMetrics(
        pairs=len(scored),
        cases=len(cases),
        samples_per_prefix=samples_per_prefix,
        truncated_pairs_excluded=truncated_pairs_excluded,
        scores=PrefixScores.mean(scored),
        by_prefix_length=[
            ByPrefixLengthMetrics(
                length=length,
                pairs_count=len(buckets[length]),
                scores=PrefixScores.mean(buckets[length]),
            )
            for length in sorted(buckets)
        ],
    )
