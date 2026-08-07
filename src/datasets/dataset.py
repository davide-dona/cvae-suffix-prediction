from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

from src.datasets.codec import DatasetCodec
from src.logs.io import read_log
from src.logs.keys import CASE_KEY
from src.paths import Split


class Events(NamedTuple):
    """The set of events composing a single trace, as the model reads it."""

    activities: torch.Tensor  # int64, [..., seq_len]
    resources: torch.Tensor  # int64, [..., seq_len]
    ts_prev: torch.Tensor  # float32, standardized time since the previous event, [..., seq_len]
    ts_start: torch.Tensor  # float32, standardized time since the case began, [..., seq_len]
    categorical_attributes: torch.Tensor  # int64, [..., seq_len, num_categorical]
    numeric_attributes: torch.Tensor  # float32, standardized, [..., seq_len, num_numeric]
    numeric_attributes_present: (
        torch.Tensor
    )  # float32 0/1, 0.0 where the log had no value [..., seq_len, num_numeric]
    length: torch.Tensor  # int64, the real, unpadded length of the run, [...], 1D

    def cut(self, index: slice | torch.Tensor) -> Events:
        """Slice every per-event field by index, updating `length`."""
        # Every field but `length`, which is recounted from what the slice kept.
        channels = [channel[index] for channel in self[:-1]]
        return Events(*channels, length=torch.tensor(data=len(channels[0]), dtype=torch.long))

    def padded(self, to: int) -> Events:
        """Pad every per-event field to `to` positions, leaving `length` unchanged.
        This allows a batch of runs to be stacked into a single tensor, with the padding masked
        out by `pad_mask`.

        Args:
            to: The width to pad to, `max_trace_length` for everything the model reads.
        Returns:
            The same events at the front of `to` positions, `length` unchanged.
        """
        # Every field but `length`, which counts real events and so survives the padding.
        return Events(
            *(
                torch.cat(
                    tensors=(
                        channel,
                        torch.zeros(
                            size=(to - channel.size(dim=0), *channel.shape[1:]), dtype=channel.dtype
                        ),
                    )
                )
                for channel in self[:-1]
            ),
            length=self.length,
        )

    def pad_mask(self) -> torch.Tensor:
        """Marks the positions that were padded out.
        Returns:
            `[batch_size, seq_len]`, True where the position holds padding.
        """
        positions = torch.arange(end=self.activities.size(dim=-1), device=self.length.device)
        return positions.unsqueeze(dim=0) >= self.length.unsqueeze(dim=1)

    def to(self, device: torch.device) -> Events:
        """Move a whole batch in one call"""
        return Events(*(field.to(device) for field in self))


class SplitTrace(NamedTuple):
    """A single trace cut into a (prefix, suffix) pair, with the remaining time from the last
    prefix event to the case's real ending."""

    # Which case of the log this was cut from. Allow to identify the original trace after
    # generation.
    case_id: str  # a `tuple[str, ...]` of `batch_size` of them once collated

    prefix: Events  # the condition: the events before the cut, no EOT
    suffix: Events  # what the decoder must produce: content, EOT, then padding

    # The standardized minutes from the last prefix event to the case's real ending, one per item.
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
    events: Events  # the case's events, unpadded
    remaining_time: (
        torch.Tensor
    )  # standardized minutes from each event to the case's real ending, [len(events)]


class TraceDataset(Dataset):
    """A PyTorch Dataset of traces, each cut into a set of (prefix, suffix) pairs.

    A case of `n` events yields `n - 1` data points, one per cut point `k = 1 .. n - 1`:
    - **prefix**: `events[:k]`
    - **suffix**: `events[k:]` followed by EOT.

    Prefixes and suffixes are both padded to `max_trace_length`.
    """

    def __init__(self, codec: DatasetCodec, *, split: Split):
        """Read one preprocessed split and cut every trace into all possible (prefix, suffix) pairs.

        Args:
            codec: The dataset codec the split was preprocessed against, which names where the
                split is as well as what its values are encoded through.
            split: Which of the three to read.
        """
        self.codec = codec
        self.max_len = codec.max_trace_length

        # Read the split and encode it whole: the same work done per event in `__getitem__`
        # would be repeated for every cut point of every case.
        split_dataset = _read_split(codec, split=split)

        events = _encode_events(codec, split_dataset)
        remaining_time = torch.from_numpy(codec.remaining_time.encode(split_dataset))

        # Group the encoded split into per-case runs.
        self._traces = _group_cases(
            split_dataset,
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

        # Every suffix is closed with an EOT event, which is the decoder's signal to stop.
        suffix.activities[suffix_len] = self.codec.activity.eot_index
        suffix.resources[suffix_len] = self.codec.resource.eot_index
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
        # Content length plus the EOT closing it, skipping the padding and tensor work
        # `__getitem__` does: sorting has no use for either when it only wants a length.
        lengths = []
        for i in range(len(self)):
            case, k = self._get_cut(i)
            lengths.append(int(case.events.length) - k + 1)
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


def _read_split(codec: DatasetCodec, *, split: Split) -> pd.DataFrame:
    """Read one preprocessed split, returning it as a DataFrame.

    Args:
        codec: The dataset codec, naming where the split is and every categorical channel's
            column.
        split: Which of the three to read.
    Returns:
        The split, one row per event.
    """
    categorical = (codec.activity, codec.resource, *codec.categorical_features)
    text_columns = {CASE_KEY: str} | {column.column: str for column in categorical}
    return read_log(codec.split_path(split), dtype=text_columns)


def _encode_events(codec: DatasetCodec, log: pd.DataFrame) -> Events:
    """Map a run of raw events to the indices and normalized floats the model consumes.

    Args:
        codec: The dataset's codec, naming every channel read here and holding the vocabulary or
            range each is encoded through.
        log: The events as a preprocessed split holds them. The whole frame, not a selection of
            it: which columns each channel reads is the codec's answer, and this is where it is
            asked.

    Returns:
        The same events as vocabulary indices and normalized channels, unpadded, so every one of
        them counts towards `length`.
    """
    numeric_attributes, numeric_attributes_present = _encode_numeric_attributes(codec, log)
    return Events(
        # `torch.tensor` rather than `from_numpy`: pandas hands back a read-only view of its
        # own block for some dtypes, which torch would wrap rather than copy.
        activities=torch.tensor(data=codec.activity.encode(log), dtype=torch.long),
        resources=torch.tensor(data=codec.resource.encode(log), dtype=torch.long),
        ts_prev=torch.from_numpy(codec.ts_prev.encode(log)),
        ts_start=torch.from_numpy(codec.ts_start.encode(log)),
        categorical_attributes=_encode_categorical_attributes(codec, log),
        numeric_attributes=numeric_attributes,
        numeric_attributes_present=numeric_attributes_present,
        length=torch.tensor(data=len(log), dtype=torch.long),
    )


def _encode_categorical_attributes(codec: DatasetCodec, log: pd.DataFrame) -> torch.Tensor:
    """Every categorical attribute channel of a run of events, packed into one index array.

    Args:
        codec: The dataset's codec, holding the feature channels and their blocks of the shared
            table.
        log: The events as a preprocessed split holds them, so a gap in a channel already carries
            the missing token preprocessing gave it.
    Returns:
        `[len(log), num_categorical]` of rows of the shared embedding table.
    """
    if not codec.categorical_features:
        return torch.zeros(size=(len(log), 0), dtype=torch.long)
    return torch.stack(
        tensors=[
            torch.tensor(data=feature.encode(log), dtype=torch.long)
            for feature in codec.categorical_features
        ],
        dim=1,
    )


def _encode_numeric_attributes(
    codec: DatasetCodec, log: pd.DataFrame
) -> tuple[torch.Tensor, torch.Tensor]:
    """Every numeric attribute channel's normalized value, and the flag saying it was there.

    Args:
        codec: The dataset's codec, holding the feature channels and their ranges.
        log: The events as rows of the log.
    Returns:
        The values and the flags, `[len(log), num_numeric]` each.
    """
    if not codec.numeric_features:
        empty = torch.zeros(size=(len(log), 0), dtype=torch.float32)
        return empty, empty.clone()
    return (
        torch.stack(
            tensors=[torch.from_numpy(feature.encode(log)) for feature in codec.numeric_features],
            dim=1,
        ),
        torch.stack(
            tensors=[torch.from_numpy(feature.present(log)) for feature in codec.numeric_features],
            dim=1,
        ),
    )


def _group_cases(
    split_dataset: pd.DataFrame,
    *,
    events: Events,
    remaining_time: torch.Tensor,
) -> list[_Trace]:
    """Group a split's already-encoded events into per-case runs.

    Args:
        split_dataset: The split, from `_read_split`; only its case column and row order are
            read here, the values themselves already encoded into `events`.
        events: The split's events, encoded whole, indexed the same as `split_dataset`.
        remaining_time: The split's standardized remaining time, indexed the same way.
    Returns:
        One `_Trace` per case of the split, each of them whole: preprocessing dropped the cases
        that do not fit `max_trace_length`, so nothing is cut short here.
    """
    cases = []
    for case_id, group in split_dataset.groupby(CASE_KEY, sort=False):
        # The case's rows as positions into the split-wide columns.
        positions = torch.from_numpy(split_dataset.index.get_indexer(group.index))
        cases.append(
            _Trace(
                case_id=str(case_id),
                events=events.cut(positions),
                remaining_time=remaining_time[positions],
            )
        )
    return cases
