from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from src.inference import Generation
from src.metrics import ScalarMetrics, mean

MINUTES_PER_DAY = 1440.0


@dataclass(frozen=True)
class AccuracyScores(ScalarMetrics):
    """The scores of one prefix's generated samples, or their mean over a set of prefixes.

    These are the free-running numbers: the model wrote every suffix scored here on its own, one
    event at a time. `Loss.remaining_time_loss` scores the same head teacher-forced, which is a
    different question and a far easier one.

    Everything here is cheap enough to compute at every validation, and everything here is
    computed on both paths, the training curve and the final report. A number only the report can
    afford belongs beside `ConformanceScores` in `conformance.py` instead.

    No defaults: a field added here must be set in `score_generation`, and a `TypeError` is how
    that is enforced.
    """

    # The Damerau-Levenshtein Similarity (DLS) is the edit distance normalized to [0, 1] and
    # inverted.
    dls_mean: float  # The mean similarity of a prefix's samples to the ground truth
    dls_point: (
        float  # z = mean(p(z | prefix), a single greedy answer, scored against the ground truth
    )

    # The share of prefixes whose true suffix is exactly among their first k samples: what k
    # suggestions are worth to a user who only needs one of them to be right. 0.0 or 1.0 for a
    # single prefix, a rate once averaged over a set of them.
    hit_rate_at_1: float
    hit_rate_at_5: float
    hit_rate_at_10: float

    # The samples read as a predictive distribution, lower being better. The score a checkpoint
    # is selected on.
    energy_score: float

    # How far apart the samples of a prefix are, and how many of them are distinct sequences.
    # Both say how much of the prefix's uncertainty z carries, not how good the model is.
    sample_diversity: float
    unique_sample_rate: float

    # Absolute error (AE) between the predicted and true remaining cycle time, in days.
    # No best since the closest of ten draws of a scalar measures how widely the head scatters
    # rather than how well it predicts.
    remaining_time_ae_mean_days: float
    remaining_time_ae_point_days: float

    # Absolute error (AE) between the predicted and true suffix length, in events.
    length_ae_mean: float
    length_ae_point: float

    # Events left after the cut point: the scale every error above is read against. A property of
    # the prefixes scored rather than of the model, so it is flat across a training run.
    suffix_length: float


def damerau_levenshtein_distance(first: Sequence[Hashable], second: Sequence[Hashable]) -> int:
    """The number of edits to turn one sequence into another, allowing transpositions of
    adjacent elements as one edit.

    Args:
        first: The first sequence.
        second: The sequence to measure it against.
    Returns:
        The number of edits, at least `abs(len(first) - len(second))` and at most
        `max(len(first), len(second))`.
    """
    # Row i, column j holds the distance between the first i and the first j elements.
    distances = [[0] * (len(second) + 1) for _ in range(len(first) + 1)]

    # Initialize the first row and column to the distance from the empty sequence.
    for i in range(len(first) + 1):
        distances[i][0] = i
    for j in range(len(second) + 1):
        distances[0][j] = j

    for i in range(1, len(first) + 1):
        for j in range(1, len(second) + 1):
            substitution_cost = 0 if first[i - 1] == second[j - 1] else 1
            distances[i][j] = min(
                distances[i - 1][j] + 1,  # delete
                distances[i][j - 1] + 1,  # insert
                distances[i - 1][j - 1] + substitution_cost,  # substitute, or match for free
            )
            # The two elements are each other's, the other way round: one edit, not two.
            if i > 1 and j > 1 and first[i - 1] == second[j - 2] and first[i - 2] == second[j - 1]:
                distances[i][j] = min(distances[i][j], distances[i - 2][j - 2] + 1)

    return distances[len(first)][len(second)]


def sequence_similarity(predicted: Sequence[Hashable], true: Sequence[Hashable]) -> float:
    """Damerau-Levenshtein similarity, the distance normalized into `[0, 1]`.

    Args:
        predicted: The generated sequence.
        true: The ground-truth sequence.
    Returns:
        1.0 for identical sequences (two empty ones included), down to 0.0 for sequences
        sharing nothing.
    """
    longest = max(len(predicted), len(true))
    if longest == 0:
        return 1.0
    return 1.0 - damerau_levenshtein_distance(predicted, true) / longest


def diversity(samples: Sequence[Sequence[Hashable]]) -> float:
    """
    How far apart a set of sequences generated for one prefix are from each other, in `[0, 1]`.

    The mean distance over every pair, which is 0.0 when a prefix's samples are all the same
    sequence (or there are fewer than two to compare). Comparing samples of one prefix against
    each other is what measures the spread `p(z | prefix)` claims the prefix leaves open.

    Args:
        samples: The sequences generated for one prefix, one per draw of z.
    Returns:
        0.0 for identical (or singleton) sample sets, up to 1.0 for sequences sharing nothing.
    """
    pairs = [
        1.0 - sequence_similarity(samples[first], samples[second])
        for first in range(len(samples))
        for second in range(first + 1, len(samples))
    ]
    return mean(pairs)


def energy_score(dls_mean: float, sample_diversity: float) -> float:
    """The samples read as a predictive distribution, lower being better.

    The mean distance to the truth, discounted by half the spread the samples already cover.
    Spread only pays off where the truth is far from a single guess, which is what makes this a
    score to minimize rather than a number to read beside `dls_mean`, neither of which having a
    target value.

    The discount stays below the distance, so the score stays at or above 0.0, only where
    `1 - DLS` obeys the triangle inequality. Dividing by the longer of two lengths does not
    guarantee that: `eb` and `bc` are 1.0 apart while both sit 1/3 from `ebc`. Such sequences
    are rare enough not to show up in a real generation, so this is normalized the way the
    reported similarity is rather than a second way that would be.
    """
    return (1.0 - dls_mean) - 0.5 * sample_diversity


def is_hit(samples: Sequence[tuple[str, ...]], truth: tuple[str, ...], *, k: int) -> float:
    """
    Whether the true sequence is among the first `k` samples, exactly.

    Args:
        samples: The activity sequences generated for one prefix, in the order they were drawn.
            A draw is independent of the ones before it, so the first `k` of them are as good a
            sample of `k` as any other choice.
        truth: The ground-truth activity sequence.
        k: How many samples to look at.
    Returns:
        1.0 if one of the first `k` samples is the true sequence, 0.0 otherwise.
    """
    return float(truth in samples[:k])


def score_generation(generation: Generation) -> AccuracyScores:
    """
    Score the suffixes generated for one prefix against the ground truth they continue.

    The one place these numbers are defined: `validate_generation` averages them over a
    validation slice while training, `evaluate_generations` over the test split afterwards, so a
    training curve and a final report measure the same thing rather than agreeing by coincidence.
    What the report adds on top of them is `conformance.py`, computed there alone.

    Args:
        generation: The model's answer for one prefix, decoded into the log's own units.
    Returns:
        The prefix's scores. A prefix with no samples scores 0.0 on everything and 1.0 on
        `energy_score`, the worst it can be, rather than looking like a perfect prediction.
    """
    samples, point, truth = generation.samples, generation.point, generation.truth

    similarities = [sequence_similarity(sample.activities, truth.activities) for sample in samples]
    dls_mean = mean(similarities)
    sample_diversity = diversity([sample.activities for sample in samples])
    sample_activities = [tuple(sample.activities) for sample in samples]
    truth_activities = tuple(truth.activities)

    return AccuracyScores(
        dls_mean=dls_mean,
        dls_point=sequence_similarity(point.activities, truth.activities),
        hit_rate_at_1=is_hit(samples=sample_activities, truth=truth_activities, k=1),
        hit_rate_at_5=is_hit(samples=sample_activities, truth=truth_activities, k=5),
        hit_rate_at_10=is_hit(samples=sample_activities, truth=truth_activities, k=10),
        energy_score=(
            energy_score(dls_mean=dls_mean, sample_diversity=sample_diversity) if samples else 1.0
        ),
        sample_diversity=sample_diversity,
        unique_sample_rate=len(set(sample_activities)) / len(samples) if samples else 0.0,
        remaining_time_ae_mean_days=mean(
            [
                abs(sample.remaining_time_minutes - truth.remaining_time_minutes)
                for sample in samples
            ]
        )
        / MINUTES_PER_DAY,
        remaining_time_ae_point_days=abs(
            point.remaining_time_minutes - truth.remaining_time_minutes
        )
        / MINUTES_PER_DAY,
        length_ae_mean=mean([float(abs(len(sample) - len(truth))) for sample in samples]),
        length_ae_point=float(abs(len(point) - len(truth))),
        suffix_length=float(len(truth)),
    )
