from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

from src.datasets.codec import Events, encode_events
from src.datasets.description import DatasetDescription
from src.logs.io import read_log
from src.logs.keys import CASE_KEY, EVENT_DELTA_KEY, REMAINING_TIME_KEY


class SplitTrace(NamedTuple):
    """A single trace cut into a (prefix, suffix) pair, with the remaining time from the last
    prefix event to the case's real ending."""

    # Which case of the log this was cut from. Allow to identify the original trace after
    # generation.
    case_id: str  # a `tuple[str, ...]` of `batch_size` of them once collated

    prefix: Events  # the condition: the events before the cut, no EOT
    suffix: Events  # what the decoder must produce: content, EOT, then padding

    # The normalized minutes from the last prefix event to the case's real ending, one per item.
    remaining_time: torch.Tensor  # float32, [] per item, [batch_size] batched

    def to(self, device: torch.device) -> SplitTrace:
        """Move a whole batch in one call"""
        return SplitTrace(
            case_id=self.case_id,
            prefix=self.prefix.to(device),
            suffix=self.suffix.to(device),
            remaining_time=self.remaining_time.to(device),
        )


@dataclass(frozen=True)
class _Trace:
    """One case of the log, encoded once and shared by every trace cut from it."""

    case_id: str  # which case of the log this is
    # whether the case was longer than `max_trace_length`, so the suffix cut from it does not
    # actually end
    truncated: bool
    events: Events  # the case's events, unpadded
    remaining_time: (
        torch.Tensor
    )  # normalized minutes from each event to the case's real ending, [len(events)]


class TraceDataset(Dataset):
    """A PyTorch Dataset of traces, each cut into a set of (prefix, suffix) pairs.

    A case of `n` events yields `n - 1` data points, one per cut point `k = 1 .. n - 1`:
    - **prefix**: `events[:k]`
    - **suffix**: `events[k:]` followed by EOT.

    Prefixes and suffixes are both padded to `max_trace_length`.
    """

    def __init__(self, description: DatasetDescription, *, split: str):
        """Read one preprocessed split and cut every trace into all possible (prefix, suffix) pairs.

        Args:
            description: The dataset description the split was preprocessed against, which names
                where the split is as well as what its values are encoded through.
            split: Which of `train`, `val`, `test` to read.
        """
        self.description = description
        self.max_len = description.max_trace_length

        # Read the split and encode it whole: the same work done per event in `__getitem__`
        # would be repeated for every cut point of every case.
        split_dataset = _read_split(description, split=split)

        events = encode_events(
            description,
            time_deltas_minutes=split_dataset[EVENT_DELTA_KEY].to_numpy(dtype=np.float32),
            log=split_dataset,
        )
        remaining_time = torch.from_numpy(
            description.remaining_time.normalize(
                split_dataset[REMAINING_TIME_KEY].to_numpy(dtype=np.float32)
            )
        )

        # Group the encoded split into per-case runs, truncated to `max_trace_length` events.
        self._traces = _group_cases(
            split_dataset,
            max_content_len=self.max_len,
            events=events,
            remaining_time=remaining_time,
        )

        # Build the list of (case index, cut point) pairs once here rather than on every
        # __getitem__ call.
        self._cuts: list[tuple[int, int]] = [
            # The k-th cut of the case at case_idx, yielding prefix[:k] and suffix[k:].
            (case_idx, k)
            for case_idx, case in enumerate(self._traces)
            # Every cut point of the case, from 1 to len - 1, yielding n - 1 traces for a case
            # of n events.
            for k in range(1, int(case.events.length))
        ]

    def _get_cut(self, i: int) -> tuple[_Trace, int]:
        """Return the trace and cut point for the i-th trace."""
        case_idx, k = self._cuts[i]
        return self._traces[case_idx], k

    def __len__(self) -> int:
        """Return the number of traces in this split."""
        return len(self._cuts)

    def __getitem__(self, i: int) -> SplitTrace:
        """Return the i-th trace, both of its runs padded to `max_trace_length`."""
        # Retrieve the case and cut point for the i-th trace
        case, k = self._get_cut(i)
        suffix_len = int(case.events.length) - k

        # The first k events of the case, padded to `max_trace_length`
        prefix = case.events.cut(slice(0, k)).padded(to=self.max_len)
        # The last len - k events of the case, padded to `max_trace_length`.
        suffix = case.events.cut(slice(k, None)).padded(to=self.max_len)

        # If the case was not truncated, append an EOT token to the suffix and increase its
        # length by 1. Adding it anyways would teach the model to stop at `max_trace_length`
        # rather than where traces really end.
        if not case.truncated:
            suffix.activities[suffix_len] = self.description.activity.eot_index
            suffix.resources[suffix_len] = self.description.resource.eot_index
            suffix = suffix._replace(length=suffix.length + 1)

        # The remaining time, read from the last prefix event to the case's real ending
        return SplitTrace(
            case_id=case.case_id,
            prefix=prefix,
            suffix=suffix,
            remaining_time=case.remaining_time[k - 1],
        )

    def length_sorted_indices(self) -> list[int]:
        """Order this split's traces by how many positions their suffix takes to decode.

        `Decoder.generate` runs a batch until every one of its rows has emitted EOT, so a batch
        of similarly-long suffixes finishes together; unsorted, almost every batch contains one
        early-cut, long-suffix straggler and decodes to nearly the cap regardless of its other
        rows. Sorting the whole split first is what lets that early exit actually save time.

        Returns:
            SplitTrace indices, ascending by suffix length. Passed as a `DataLoader` sampler.
        """
        # Content length plus an EOT if the case has one, skipping the padding and tensor work
        # `__getitem__` does: sorting has no use for either when it only wants a length.
        lengths = []
        for i in range(len(self)):
            case, k = self._get_cut(i)
            content_len = int(case.events.length) - k
            lengths.append(content_len if case.truncated else content_len + 1)
        return sorted(range(len(self)), key=lengths.__getitem__)


def fixed_subset(dataset: Dataset, *, size: int, generator: torch.Generator) -> Dataset:
    """A random slice of `dataset`, or the whole of it if it is already no bigger.

    Drawn once, so every validation of a run reads the same traces and two of its points differ
    because the model moved rather than because the sample did.

    Args:
        dataset: The split to take from.
        size: How many items to keep.
        generator: The run's seeded generator.
    Returns:
        The slice, as a `Subset` the loaders can be built on directly.
    """
    # If the dataset is smaller than the requested size, just return it whole
    if len(dataset) <= size:
        return dataset
    # Otherwise, draw a random slice of the requested size and return it as a Subset
    indices = torch.randperm(n=len(dataset), generator=generator)[:size]
    return Subset(dataset=dataset, indices=indices.tolist())


def _read_split(description: DatasetDescription, *, split: str) -> pd.DataFrame:
    """Read one preprocessed split, returning it as a DataFrame.

    Args:
        description: The dataset description, naming where the split is and every categorical
            channel's column.
        split: Which of `train`, `val`, `test` to read.
    Returns:
        The split, one row per event.
    """
    categorical = (description.activity, description.resource, *description.categorical_features)
    text_columns = {CASE_KEY: str} | {column.column: str for column in categorical}
    return read_log(description.split_path(split), dtype=text_columns)


def _group_cases(
    split_dataset: pd.DataFrame,
    *,
    max_content_len: int,
    events: Events,
    remaining_time: torch.Tensor,
) -> list[_Trace]:
    """Group a split's already-encoded events into per-case runs, truncated to `max_content_len`.

    Args:
        split_dataset: The split, from `_read_split`; only its case column and row order are
            read here, the values themselves already encoded into `events`.
        max_content_len: How many events of a case to keep; a longer case is truncated.
        events: The split's events, encoded whole, indexed the same as `split_dataset`.
        remaining_time: The split's normalized remaining time, indexed the same way.
    Returns:
        One `_Trace` per case of the split.
    """
    cases = []
    for case_id, group in split_dataset.groupby(CASE_KEY, sort=False):
        # The case's rows as positions into the split-wide columns, cut to the length limit.
        positions = torch.from_numpy(split_dataset.index.get_indexer(group.index)[:max_content_len])
        cases.append(
            _Trace(
                case_id=str(case_id),
                truncated=len(group) > max_content_len,
                events=events.cut(positions),
                remaining_time=remaining_time[positions],
            )
        )
    return cases
