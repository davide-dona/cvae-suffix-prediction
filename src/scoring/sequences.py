from typing import Hashable, Sequence


def mean(values: Sequence[float]) -> float:
    """The mean of `values`, or 0.0 if there are none."""
    return sum(values) / len(values) if values else 0.0


def damerau_levenshtein_distance(first: Sequence[Hashable], second: Sequence[Hashable]) -> int:
    """
    The number of insertions, deletions, substitutions and transpositions between two sequences.

    Transpositions count as one edit rather than two, which is what distinguishes this from a
    plain Levenshtein distance and why it is the distance suffix prediction is scored with: two
    activities in the wrong order is a lighter mistake than two unrelated activities.

    This is the optimal-string-alignment variant, where no substring is edited more than once.
    Sequences here are at most `max_seq_len` events long, so the quadratic table costs nothing.

    Args:
        first: The first sequence.
        second: The sequence to measure it against.
    Returns:
        The number of edits, at least `abs(len(first) - len(second))` and at most
        `max(len(first), len(second))`.
    """
    # Row i, column j holds the distance between the first i and the first j elements. The
    # extra row and column are the empty prefixes, whose distance is the length of the other.
    distances = [[0] * (len(second) + 1) for _ in range(len(first) + 1)]
    for i in range(len(first) + 1):
        distances[i][0] = i
    for j in range(len(second) + 1):
        distances[0][j] = j

    for i in range(1, len(first) + 1):
        for j in range(1, len(second) + 1):
            substitution_cost = 0 if first[i - 1] == second[j - 1] else 1
            distances[i][j] = min(
                distances[i - 1][j] + 1,                       # delete
                distances[i][j - 1] + 1,                       # insert
                distances[i - 1][j - 1] + substitution_cost,   # substitute, or match for free
            )
            # The two elements are each other's, the other way round: one edit, not two.
            if (
                i > 1 and j > 1
                and first[i - 1] == second[j - 2]
                and first[i - 2] == second[j - 1]
            ):
                distances[i][j] = min(distances[i][j], distances[i - 2][j - 2] + 1)

    return distances[len(first)][len(second)]


def sequence_similarity(predicted: Sequence[Hashable], true: Sequence[Hashable]) -> float:
    """
    Damerau-Levenshtein similarity, the distance normalized into `[0, 1]`.

    Dividing by the longer of the two lengths is what makes the score comparable across cut
    points: an edit costs more on a short suffix than on a long one, and a suffix of the wrong
    length is penalized by the edits it takes to reach the right one.

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
    """
    The samples read as a predictive distribution, lower being better.

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

