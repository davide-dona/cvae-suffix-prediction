from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import OSA

# Start of the Unicode private use area, where the activity codes are drawn from.
_FIRST_CODE = 0xE000


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


def distances(
    queries: Sequence[Sequence[Hashable]],
    choices: Sequence[Sequence[Hashable]],
    *,
    dtype: type[np.floating] = np.float32,
) -> np.ndarray:
    """Measure every sequence of one set against every sequence of another.

    Every caller is already inside the evaluation's process pool, so the pairs are walked on the
    calling thread: a thread per core per process would oversubscribe the machine.

    Args:
        queries: The sequences to measure, one row each. Either encoded suffixes, where a sequence
            is a string, or raw activity names, where it is a list of them.
        choices: The sequences to measure them against, one column each.
        dtype: What to accumulate in. The default halves a matrix that can run to hundreds of
            megabytes; a caller measuring a handful of sequences has no such matrix and may ask for
            `np.float64` instead.
    Returns:
        `[len(queries), len(choices)]`, holding `1 - sequence_similarity` for each pair: 0.0 for
        two identical sequences, up to 1.0 for two sharing nothing.
    """
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
    good value of its own, which is why the two are only ever read as a pair.

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
