from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import paths
from src.identity import read_vocabulary, with_vocabulary
from src.logs.keys import (
    ACTIVITY_KEY,
    CASE_KEY,
    CYCLE_TIME_KEY,
    REMAINING_TIME_KEY,
    UNK_TOKEN,
    Split,
)
from src.suffixes import ActivityCodes, diversity

_SCHEMA = pa.schema(
    [
        # The prefix, the activities it runs, one character each
        ('prefix', pa.large_string()),
        # List of the distinct continuations observed after that prefix
        ('suffixes', pa.list_(pa.large_string())),
        # How many occurrences each of them accounts for, in the same order
        ('weights', pa.list_(pa.int32())),
        # Minutes left at the cut, one per occurrence
        ('remaining_times', pa.list_(pa.float32())),
        ('diversity', pa.float32()),
        ('cycle_time_activities', pa.large_string()),
        ('cycle_times', pa.list_(pa.list_(pa.float32()))),
    ]
)
_COMPRESSION = 'zstd'


@dataclass(frozen=True, slots=True)
class Continuations:
    """Every continuation one prefix was observed to take.
    It defines the reference distribution a generated suffix is compared against.
    """

    # The distinct continuations observed after the prefix
    suffixes: tuple[str, ...]
    # How many occurrences each of them accounts for, in the same order
    weights: np.ndarray
    # Minutes left at the cut, one per occurrence
    remaining_times: np.ndarray
    # How far apart two occurrences of this prefix ran, over the suffixes they took: the mean
    # pairwise distance `diversity` measures, weighted so two occurrences of one suffix sit at 0.
    # Reported as `reference_diversity`, the scale a model's own `sample_diversity` is read
    # against. Reduced here rather than at scoring time because it is a property of the log, the
    # same for every model of it.
    diversity: float
    # The `CYCLE_TIME_KEY` gaps observed before each activity, in minutes, keyed by the character
    # `suffixes` spells that activity with. The same quantity the decoder's `cycle_time` head
    # predicts, so the two sides of `activity_time_wasserstein_days` are comparable. A cycle time
    # belongs to the activity it precedes rather than to the position it fell at, so the occurrences
    # are pooled under their activities instead of being kept as runs: a pool is all a score reads.
    cycle_times: dict[str, np.ndarray]

    @property
    def occurrences(self) -> float:
        """How many times the prefix was observed."""
        return float(self.weights.sum())


class ContinuationIndex:
    """Every continuation one held-out split takes, keyed by the prefix it leaves behind.

    Held as the file's own columns, one entry per indexed prefix, since a lookup reads a single row
    and a split runs to hundreds of thousands of them. Reach an index through `of`, which builds one
    from a split, or `read`, which reads back what `write` left beside the splits.
    """

    def __init__(
        self,
        *,
        vocabulary: tuple[str, ...],
        rows: dict[str, int],
        suffixes: list[list[str]],
        weights: list[list[int]],
        remaining_times: list[list[float]],
        cycle_time_activities: list[str],
        cycle_times: list[list[list[float]]],
        diversity: list[float],
    ) -> None:
        """The file's columns, and which row each prefix sits at. Private: use `of` or `read`."""
        self._vocabulary = vocabulary
        self._rows = rows
        self._suffixes = suffixes
        self._weights = weights
        self._remaining_times = remaining_times
        self._cycle_time_activities = cycle_time_activities
        self._cycle_times = cycle_times
        self._diversity = diversity

    @classmethod
    def of(
        cls,
        rows: pd.DataFrame,
        *,
        vocabulary: Collection[str],
        names: Sequence[str],
    ) -> Self:
        """Index every continuation one held-out split takes.

        The reference distribution a generated suffix is compared against is one held-out split and
        nothing else: an out-of-time sample of `p(suffix | prefix)`, so a model that memorized the
        train split gains nothing from it and the reference is not drawn from the older regime the
        split's own separation exists to keep apart. Every cut point of every case contributes,
        `min_prefix_len` included, since that bound governs what a model may be asked rather than
        what the log was observed to do.

        Both held-out splits are indexed, and which one is read is the reader's choice: evaluation
        scores against the test split, while training selects checkpoints against the validation
        split, since selecting on the test split's continuations would fold the held-out set into
        what gets kept.

        The split is read through the train split's vocabulary, an activity missing from it becoming
        UNK, which is the only name generation can give it: a model can emit no activity it was
        never shown, and the generations name a prefix and its ground truth the same way. Keying the
        index on the raw names instead would leave every prefix of an out-of-time activity
        unlookupable, and every reference suffix holding one unmatchable by construction.

        Args:
            rows: The split to index, as preprocessing holds it, sorted by case and by timestamp.
            vocabulary: The activity names the train split holds, from `codec.activity.vocab`.
            names: Every name the activity channel can decode to, in row order, from
                `codec.activity.names`. Seeds the codebook, so a suffix is spelled here exactly as a
                generations file spells it and neither side ever codes a name on the fly.
        Returns:
            The index, ready to be queried or written.
        """
        known = set(vocabulary)
        seen = rows.assign(
            **{
                ACTIVITY_KEY: rows[ACTIVITY_KEY].where(
                    cond=rows[ACTIVITY_KEY].isin(known), other=UNK_TOKEN
                )
            }
        )

        codes = ActivityCodes.of(names)
        cases = [
            (
                codes.encode(events[ACTIVITY_KEY]),
                events[REMAINING_TIME_KEY].tolist(),
                events[CYCLE_TIME_KEY].tolist(),
            )
            for _, events in seen.groupby(CASE_KEY, sort=False)
        ]

        # Every cut point of every case, grouped under the prefix it leaves behind. A prefix's
        # continuations are counted as they arrive, so a suffix the log took twice is one entry
        # weighing two; its remaining times are kept one per occurrence, and the cycle time before
        # each of its events is pooled under that event's own activity. The first event of a case
        # carries no cycle time, and it is never in a suffix either, since a cut leaves at least one
        # event behind it.
        grouped: dict[str, tuple[dict[str, int], list[float], dict[str, list[float]]]] = {}
        for activities, remaining_times, deltas in cases:
            for cut in range(1, len(activities)):
                observed, minutes, cycle_times = grouped.setdefault(activities[:cut], ({}, [], {}))
                suffix = activities[cut:]
                observed[suffix] = observed.get(suffix, 0) + 1
                minutes.append(remaining_times[cut - 1])
                for activity, cycle_time in zip(suffix, deltas[cut:], strict=True):
                    cycle_times.setdefault(activity, []).append(cycle_time)

        return cls(
            vocabulary=codes.vocabulary,
            rows={prefix: row for row, prefix in enumerate(grouped)},
            suffixes=[list(observed) for observed, _, _ in grouped.values()],
            weights=[list(observed.values()) for observed, _, _ in grouped.values()],
            remaining_times=[minutes for _, minutes, _ in grouped.values()],
            cycle_time_activities=[''.join(cycle_times) for _, _, cycle_times in grouped.values()],
            cycle_times=[list(cycle_times.values()) for _, _, cycle_times in grouped.values()],
            diversity=[
                diversity(tuple(observed), weights=list(observed.values()))
                for observed, _, _ in grouped.values()
            ],
        )

    @classmethod
    def read(cls, dataset: str, split: Split) -> Self:
        """Read back the index preprocessing wrote.

        Args:
            dataset: The dataset whose index to read, from where preprocessing wrote it.
            split: Which split's continuations to read. Evaluation scores against `TEST`, and
                training selects checkpoints against `VAL`.
        Returns:
            The index that split was written as.
        """
        table = pq.read_table(paths.CONTINUATIONS.require(dataset=dataset, split=split))
        prefixes = table.column('prefix').to_pylist()
        return cls(
            vocabulary=read_vocabulary(table.schema),
            rows={prefix: row for row, prefix in enumerate(prefixes)},
            suffixes=table.column('suffixes').to_pylist(),
            weights=table.column('weights').to_pylist(),
            remaining_times=table.column('remaining_times').to_pylist(),
            cycle_time_activities=table.column('cycle_time_activities').to_pylist(),
            cycle_times=table.column('cycle_times').to_pylist(),
            diversity=table.column('diversity').to_pylist(),
        )

    def write(self, *, dataset: str, split: Split) -> Path:
        """Write the index beside the splits, for `read` to load it back.

        Args:
            dataset: The dataset the split came from, naming where the index goes.
            split: Which split this is, naming the file written.
        Returns:
            The file written.
        """
        # `_rows` was built by enumerating the prefixes in order, so its keys are the prefix
        # column and reading them back out is what keeps every row's index pointing at its own data.
        table = pa.table(
            {
                'prefix': list(self._rows),
                'suffixes': self._suffixes,
                'weights': self._weights,
                'remaining_times': self._remaining_times,
                'diversity': self._diversity,
                'cycle_time_activities': self._cycle_time_activities,
                'cycle_times': self._cycle_times,
            },
            schema=with_vocabulary(_SCHEMA, self._vocabulary),
        )
        path = paths.CONTINUATIONS.prepare(dataset=dataset, split=split)
        pq.write_table(table, path, compression=_COMPRESSION)
        return path

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """The activity names this index codes through, in code order.

        Both this and the generations file are seeded from `codec.activity.names`, so the two agree
        by construction and a caller compares them to catch a file built against an older
        preprocessing before it scores a single prefix.
        """
        return self._vocabulary

    @property
    def prefixes(self) -> int:
        """How many distinct prefixes are indexed."""
        return len(self._rows)

    @property
    def occurrences(self) -> int:
        """How many cut points those prefixes cover, counting a prefix once per occurrence."""
        return sum(len(minutes) for minutes in self._remaining_times)

    def continuations(self, prefix: str) -> Continuations:
        """Every continuation observed after one prefix.

        Args:
            prefix: The prefix's activities, one character each, in order, as the generations file
                carries them and on the same scale this index is held on.
        Returns:
            The prefix's observed continuations, their weights, the minutes left at each of its
            occurrences and the spread of the set.
        Raises:
            KeyError: If the split never ran that prefix, which means the generations were written
                against a different preprocessing of this dataset than the index was built from.
        """
        row = self._rows.get(prefix)
        if row is None:
            raise KeyError(
                f'a prefix of {len(prefix)} events is in the generations but not in the '
                'continuation index: the two were built from different preprocessings of this '
                'dataset. Rerun pipelines.preprocess, then pipelines.generate.'
            )
        return Continuations(
            suffixes=tuple(self._suffixes[row]),
            weights=np.asarray(self._weights[row], dtype=np.float64),
            remaining_times=np.asarray(self._remaining_times[row], dtype=np.float64),
            cycle_times={
                activity: np.asarray(observed, dtype=np.float64)
                for activity, observed in zip(
                    self._cycle_time_activities[row], self._cycle_times[row], strict=True
                )
            },
            diversity=float(self._diversity[row]),
        )
