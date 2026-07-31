from typing import Sequence


def damerau_levenshtein_distance(first: Sequence[str], second: Sequence[str]) -> int:
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


def sequence_similarity(predicted: Sequence[str], true: Sequence[str]) -> float:
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
