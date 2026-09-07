from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import OSA
from scipy.spatial.distance import cdist

# Start of the Unicode private use area, where the activity codes are drawn from.
_FIRST_CODE = 0xE000

# What a bigram is padded with, so that a suffix of n activities yields n + 1 pairs. Both sit
# below the private use area the activity codes are drawn from, so neither can be an activity.
_START = '\x02'
_END = '\x03'


class SuffixMetric(StrEnum):
    """How far apart two suffixes are held to be.

    Three, because each charges for a different kind of wrongness. `DLS` is graded on positions, so
    a suffix one edit from another scores better than one sharing nothing with it. `EXACT` reads a
    suffix as an atom and charges the same for a near miss as for a wrong answer. `BIGRAM` charges
    for the ordered pairs a suffix holds and how many times it holds each, which is the ordering
    and the loop counts a process constrains rather than the positions they fell at.

    All three run over `[0, 1]`, so a score reading one of them reads them all on one scale. They
    do not all make `energy_score` proper, though, which needs a distance of negative type:
    `EXACT` is the discrete metric and `BIGRAM` the Jaccard distance, both of which are, and
    neither violates it on any draw set measured. `DLS` is a normalized edit distance and is not:
    it violates negative type on about 45% of real draw sets, where a distribution that drops part
    of its support can beat the true one by around half a percent of the score. It is reported
    because it is the scale the rest of the project reads on and because it is the sample-side
    counterpart of `dls_mean`, not because it settles the question the other two settle.
    """

    DLS = 'dls'  # 1 - the normalized Damerau-Levenshtein (OSA) similarity
    EXACT = 'exact'  # 1 for any two suffixes that are not the same
    BIGRAM = 'bigram'  # Multiset Jaccard distance over padded activity pairs


@dataclass(slots=True)
class ActivityCodes:
    """Map each activity name to a character. A suffix becomes a string, rather than a sequence of
    objects, making it cheaper to hold and cheaper to measure against another.
    """

    _codes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def of(cls, activities: Sequence[str]) -> 'ActivityCodes':
        """Seed a codebook from activity names already in code order.

        Args:
            activities: The names, in the order their codes were handed out, as `vocabulary`
                returns them.
        Returns:
            A codebook giving each of them the code it had, and the next code to anything else.
        """
        return cls({activity: chr(_FIRST_CODE + code) for code, activity in enumerate(activities)})

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """The activity names in code order, which is what seeds `of` back into this codebook."""
        return tuple(self._codes)

    @property
    def codes(self) -> Mapping[str, str]:
        """Each activity name to the character it is spelled with, read-only.

        For a caller that has to look a name up without handing out a code to one it has never
        seen, which `encode` would: a constraint naming an activity the log never ran must stay
        unmatchable rather than quietly joining the codebook and desyncing it from the file it was
        seeded from.
        """
        return MappingProxyType(self._codes)

    def encode(self, activities: Sequence[str]) -> str:
        """Encode one suffix, giving each activity not seen before the next code point.

        Args:
            activities: The suffix's activity names, in order.
        Returns:
            One character per activity. The empty string for an empty suffix, which a model does
            generate and which sits at distance 1.0 from every suffix holding an event.
        """
        return ''.join(
            self._codes.setdefault(activity, chr(_FIRST_CODE + len(self._codes)))
            for activity in activities
        )


def sequence_similarity(predicted: Sequence[Hashable], true: Sequence[Hashable]) -> float:
    """Damerau-Levenshtein similarity, the edit distance normalized into `[0, 1]`.

    Args:
        predicted: The generated sequence.
        true: The ground-truth sequence.
    Returns:
        1.0 for identical sequences (two empty ones included), down to 0.0 for sequences
        sharing nothing.
    """
    return OSA.normalized_similarity(predicted, true)


def bigrams(sequence: Sequence[Hashable]) -> Counter[tuple[Hashable, Hashable]]:
    """The ordered pairs a sequence holds, and how many times it holds each.

    Padded at both ends, so a sequence of `n` elements yields `n + 1` pairs rather than `n - 1`.
    Without the padding a sequence of one element would hold no pair at all and so would sit at
    distance 0 from every other sequence of one, and the empty sequence would sit at distance 0
    from itself and from nothing else measurably.

    Args:
        sequence: An encoded suffix, where an element is a character, or a list of raw activity
            names, where it is a name. Neither sentinel can collide with an activity: both sit
            below the private use area `ActivityCodes` draws from, and a raw name is a string of
            more than one character.
    Returns:
        Each pair to the number of times it occurs. Never empty: the empty sequence yields the one
        pair the two sentinels make.
    """
    padded = (_START, *sequence, _END)
    return Counter(zip(padded[:-1], padded[1:], strict=True))


def _exact_distances(
    queries: Sequence[Sequence[Hashable]],
    choices: Sequence[Sequence[Hashable]],
    *,
    dtype: type[np.floating],
) -> np.ndarray:
    """Measure every pair on the discrete metric: 0 for two equal sequences, 1 for anything else.

    Args:
        queries: The sequences to measure, one row each.
        choices: The sequences to measure them against, one column each.
        dtype: What to hold the result in.
    Returns:
        `[len(queries), len(choices)]` of 0.0 and 1.0.
    """
    # Tuples rather than the sequences themselves, so a string and a list of the same activities
    # compare as the sequences they are and `!=` never walks a NumPy array elementwise.
    left = np.empty(len(queries), dtype=object)
    left[:] = [tuple(sequence) for sequence in queries]
    right = np.empty(len(choices), dtype=object)
    right[:] = [tuple(sequence) for sequence in choices]
    return np.not_equal.outer(left, right).astype(dtype)


def _bigram_distances(
    queries: Sequence[Sequence[Hashable]],
    choices: Sequence[Sequence[Hashable]],
    *,
    dtype: type[np.floating],
) -> np.ndarray:
    """Measure every pair on the multiset Jaccard distance over their bigrams.

    `1 - |A n B| / |A u B|` on multisets, so a pair occurring twice in one sequence and once in the
    other is shared once and unioned twice: the loop counts a process constrains are charged for
    where a set of bigrams would flatten them.

    Read off the L1 distance between the two count vectors rather than by walking the pairs.
    `sum(min(a, b))` is `(|A| + |B| - L1) / 2` for counts, which puts the union at
    `(|A| + |B| + L1) / 2` and the whole distance at `2 * L1 / (|A| + |B| + L1)`: one call into
    C rather than a Python row at a time.

    Args:
        queries: The sequences to measure, one row each.
        choices: The sequences to measure them against, one column each.
        dtype: What to hold the result in.
    Returns:
        `[len(queries), len(choices)]`, 0.0 for two sequences holding the same pairs as often as
        each other, up to 1.0 for two sharing none.
    """
    counted = [bigrams(sequence) for sequence in (*queries, *choices)]
    vocabulary = {pair: column for column, pair in enumerate({pair for c in counted for pair in c})}
    # Dense over the pairs these two sets happen to hold, which is what keeps this to a prefix's
    # draws rather than to a whole log's continuations.
    matrix = np.zeros((len(counted), len(vocabulary)), dtype=np.float64)
    for row, held in enumerate(counted):
        for pair, count in held.items():
            matrix[row, vocabulary[pair]] = count

    left, right = matrix[: len(queries)], matrix[len(queries) :]
    l1 = cdist(left, right, metric='cityblock')
    sizes = left.sum(axis=1)[:, None] + right.sum(axis=1)[None, :]
    # The padding puts at least one pair in every sequence, so the union is never 0.
    return (2.0 * l1 / (sizes + l1)).astype(dtype)


def distances(
    queries: Sequence[Sequence[Hashable]],
    choices: Sequence[Sequence[Hashable]],
    *,
    metric: SuffixMetric = SuffixMetric.DLS,
    dtype: type[np.floating] = np.float32,
) -> np.ndarray:
    """Measure every sequence of one set against every sequence of another.

    Every caller is already inside the evaluation's process pool, so the pairs are walked on the
    calling thread: a thread per core per process would oversubscribe the machine.

    Args:
        queries: The sequences to measure, one row each. Either encoded suffixes, where a sequence
            is a string, or raw activity names, where it is a list of them.
        choices: The sequences to measure them against, one column each.
        metric: How far apart two sequences are held to be. `DLS` is what a transport cost and the
            per-prefix similarities read on; the other two are read by `energy_score` alone, and
            `BIGRAM` holds a dense row over the bigrams both sets hold, so it is meant for a
            prefix's draws rather than for a whole log's continuations.
        dtype: What to accumulate in. The default halves a matrix that can run to hundreds of
            megabytes; a caller measuring a handful of sequences has no such matrix and may ask for
            `np.float64` instead.
    Returns:
        `[len(queries), len(choices)]`, holding the distance of each pair: 0.0 for two identical
        sequences, up to 1.0 for two sharing nothing.
    """
    if metric is SuffixMetric.EXACT:
        return _exact_distances(queries, choices, dtype=dtype)
    if metric is SuffixMetric.BIGRAM:
        return _bigram_distances(queries, choices, dtype=dtype)

    similarities = process.cdist(
        queries=queries,
        choices=choices,
        scorer=OSA.normalized_similarity,
        dtype=dtype,
    )
    # In place: the matrix is already the largest thing here, and a second copy of it is what a
    # blocked walk exists to avoid.
    return np.subtract(1.0, similarities, out=similarities)


# How many rows of the pairwise matrix `diversity` holds at once.
_SPREAD_MATRIX_SIZE = 256


def diversity(
    sequences: Sequence[Sequence[Hashable]],
    *,
    weights: Sequence[float] | None = None,
) -> float:
    """How far apart two draws of one set of sequences are from each other, in `[0, 1]`.

    The mean distance over every ordered pair of two distinct draws. A weighted set is a set of
    distinct sequences standing for that many draws, so a sequence drawn twice is twice as likely
    to be picked and the pair it makes with itself sits at distance 0.

    Read twice per prefix, on the same scale both times: over a model's draws it is
    `sample_diversity`, and over the continuations a log took after one prefix it is
    `reference_diversity`, which is the spread `sample_diversity` is judged against. Neither has a
    good value of its own, which is why the two are only ever read as a pair. It is the same
    `E[d(X, X')]` that `energy_score` subtracts, taken over a set that can run to thousands rather
    than over one prefix's draws, which is why it walks in blocks where that one holds a matrix.

    Args:
        sequences: The distinct sequences, either encoded suffixes or raw activity names.
        weights: How many draws each of them stands for, in the same order, or `None` for one
            each.
    Returns:
        0.0 below two draws, or where every draw is the same sequence, up to 1.0 for sequences
        sharing nothing.
    """
    counts = (
        np.ones(len(sequences), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    draws = counts.sum()
    if draws < 2 or len(sequences) < 2:
        return 0.0

    # Walked in blocks: the full matrix of a prefix the log ran thousands of times is the largest
    # thing this would hold, and only one block of its rows is needed at a time.
    total = 0.0
    for first in range(0, len(sequences), _SPREAD_MATRIX_SIZE):
        block = sequences[first : first + _SPREAD_MATRIX_SIZE]
        pairs = distances(queries=block, choices=sequences, dtype=np.float64)
        total += float(counts[first : first + len(block)] @ pairs @ counts)
    return total / (draws * (draws - 1.0))


def energy_score(
    sequences: Sequence[Sequence[Hashable]],
    truth: Sequence[Hashable],
    *,
    weights: Sequence[float] | None = None,
    metric: SuffixMetric = SuffixMetric.DLS,
) -> float:
    """`E[d(X, y)] - 0.5 * E[d(X, X')]` over a set of draws, in `[-0.5, 1]`.

    CRPS generalized off the real line. Averaged over the observations, a forecast `p` sits
    `-0.5 * (p - q)' D (p - q)` from the score the truth `q` gets, so the truth wins exactly when
    the distance is of negative type, which `SuffixMetric` records for each of the three. Where it
    holds, the score is not won by putting every draw on one sequence the way a mean distance to a
    single observation is: collapsing takes the second term to 0 and leaves the first, which a
    spread that covers the truth beats. It is exactly `d(X, y)` where the draws are one value,
    which is the same relation `crps` has with an absolute error.

    Args:
        sequences: The distinct sequences drawn, either encoded suffixes or raw activity names.
        truth: The one sequence that was observed.
        weights: How many draws each distinct sequence stands for, or `None` for one each.
        metric: Which distance the score is read on. The three answer different questions and are
            reported side by side rather than as one number, and only two of them make the score
            proper.
    Returns:
        The score, lower being better. 1.0 where nothing was drawn, which is the worst any of the
        three metrics can give rather than the 0.0 an empty mean would read as a perfect score.
    """
    if not len(sequences):
        return 1.0
    counts = (
        np.ones(len(sequences), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    draws = counts.sum()
    if draws <= 0.0:
        return 1.0

    # Both terms come off one matrix over the draws and the truth together: its last column is
    # every distance to the truth and the block above-left of it is every distance between two
    # draws. A prefix draws at most a few hundred suffixes, so the matrix is small enough to hold
    # whole, which is what `diversity` walks in blocks to avoid over a whole log's continuations.
    pairs = distances(
        queries=(*sequences, truth),
        choices=(*sequences, truth),
        metric=metric,
        dtype=np.float64,
    )
    accuracy = float(counts @ pairs[:-1, -1]) / draws
    if draws < 2.0 or len(sequences) < 2:
        return accuracy
    # Ordered pairs of two distinct draws, so a suffix drawn twice pairs with itself at distance 0
    # and the estimator stays unbiased, exactly as `diversity` takes it.
    spread = float(counts @ pairs[:-1, :-1] @ counts) / (draws * (draws - 1.0))
    return accuracy - 0.5 * spread
