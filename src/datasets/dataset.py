from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

from src.configs.schema import DataConfig
from src.datasets.codec import EncodedSequence, encode_events
from src.datasets.description import DatasetDescription
from src.logs.io import read_log
from src.logs.keys import CASE_KEY, EVENT_DELTA_KEY, REMAINING_TIME_KEY


class EncodedEvents(NamedTuple):
    """One sequence of events as the model reads it: vocabulary indices and normalized floats,
    padded to `max_trace_length`. A prefix or suffix cut from a case.

    The same fields as `codec.EncodedSequence`, one stage on: those are numpy arrays of one
    trace's real length, these are the padded tensors a batch is collated from.
    """
    activities: torch.Tensor  # int64, [max_trace_length] per item, [batch_size, max_trace_length] batched
    resources: torch.Tensor
    time_deltas: torch.Tensor  # float32 in [0, 1]
    feature_categories: torch.Tensor  # int64, [max_trace_length, num_categorical] per item
    feature_values: torch.Tensor      # float32 in [0, 1], [max_trace_length, num_numeric] per item
    feature_present: torch.Tensor     # float32 0/1, 0.0 where the log had no value

    def to(self, device: torch.device) -> "EncodedEvents":
        """Move a whole batch in one call"""
        return EncodedEvents(*(field.to(device) for field in self))

class SuffixItem(NamedTuple):
    """One (prefix, suffix) pair, padded to `max_trace_length` and ready for the model to consume."""
    pair_index: torch.Tensor       # which pair of the dataset this is; used to trace a prediction back to its case and cut point

    prefix: EncodedEvents          # the condition: the events before the cut, no EOT
    prefix_len: torch.Tensor       # real events in `prefix`, the rest being padding
    
    suffix: EncodedEvents          # what the decoder must produce: content, EOT, then padding
    suffix_len: torch.Tensor       # real positions in `suffix`, EOT included where there is one

    # The decoder input is the suffix's activities shifted one step right, with SOS in position 0. 
    # Used for teacher forcing, so the decoder sees the ground truth at every step rather than its own predictions.
    decoder_input: torch.Tensor

    # The normalized minutes from the last prefix event to the case's real ending, one per item.
    remaining_time: torch.Tensor  # float32, [] per item, [batch_size] batched

    def to(self, device: torch.device) -> "SuffixItem":
        """Move a whole batch in one call"""
        return SuffixItem(*(field.to(device) for field in self))


@dataclass(frozen=True)
class _Trace:
    """One trace of the log, encoded once and shared by every (prefix, suffix) pair cut from it."""
    case_id: str                # which case of the log this is
    truncated: bool             # whether the case was longer than `max_trace_length`, so the suffix cut from it does not actually end
    events: EncodedSequence     # the events of the case, encoded once and shared by every (prefix, suffix) pair cut from it
    remaining_time: np.ndarray  # normalized minutes from each event to the case's real ending, shape [len(events)]


@dataclass(frozen=True)
class PairInfo:
    """The information needed to trace a generated suffix back to its case and cut point, so it can be evaluated against the ground truth."""
    case_id: str
    prefix_len: int
    # Whether the gt case was longer than `max_trace_length`, so the suffix cut from it does not actually end. 
    truncated: bool


class SuffixDataset(Dataset):
    """A PyTorch Dataset of (prefix, suffix) pairs, cut from every case of one split.

    A case of `n` events yields `n - 1` pairs, one per cut point `k = 1 .. n - 1`:
    - **prefix**: `events[:k]`
    - **suffix**: `events[k:]` followed by EOT.

    Only the (case, cut point) index is materialized; the padded tensors are built on demand, so
    enumerating every cut point stays cheap even on the larger logs.
    
    Prefixes and suffixes are both padded to `max_trace_length`, which is the only length
    knob: a prefix holds at most `n - 1` events and a suffix at most `n - 1` events plus
    EOT, so neither can exceed it. Cases longer than that are truncated to their first
    `max_trace_length` events, and the suffixes cut from them carry no EOT (see
    `_encode_suffix`).
    """

    def __init__(
        self, data_config: DataConfig, *, split: str, description: DatasetDescription
    ):
        """Read one preprocessed split and cut every case of it into (prefix, suffix) pairs.

        Args:
            data_config: The `data` section, naming the dataset directory.
            split: Which of `train`, `val`, `test` to read.
            description: The dataset description the split was preprocessed against, which is
                also what its values are encoded through.
        """
        self.description = description
        self.max_len = description.max_trace_length
        self._num_categorical = len(description.categorical_features)
        self._num_numeric = len(description.numeric_features)

        # Group the split into per-case traces, truncated to max_len events, with their content
        # mapped to indices once here rather than on every __getitem__ call.
        self._traces = _group_traces(
            _read_split(data_config, split=split, description=description),
            self.max_len,
            description,
        )

        # Build the list of (case index, cut point) pairs once here rather than on every __getitem__ call.
        self._pairs: list[tuple[int, int]] = [
            (trace_idx, k)  # The k-th cut of the trace at trace_idx, yielding prefix[:k] and suffix[k:].
            for trace_idx, trace in enumerate(self._traces)
            for k in range(1, len(trace.events))  # Every cut point of the trace, from 1 to len - 1, yielding n - 1 pairs for a trace of n events.
        ]

    def _get_pair(self, i: int) -> tuple[_Trace, int]:
        """Return the trace and cut point for the i-th (prefix, suffix) pair."""
        trace_idx, k = self._pairs[i]
        return self._traces[trace_idx], k
    
    def __len__(self) -> int:
        """Return the number of (prefix, suffix) pairs in this split."""
        return len(self._pairs)

    def __getitem__(self, i: int) -> SuffixItem:
        """Return the i-th (prefix, suffix) pair, padded to `max_trace_length`."""
        # Retrieve the trace and cut point for the i-th pair
        trace, k = self._get_pair(i)

        # The cut itself, made here and only here: the first `k` events are the condition, and
        # everything from `k` on is what the model is asked to produce. The prefix needs no
        # encoding beyond padding - it is a cut, not a finished trace, so it carries no EOT.
        decoder_input, suffix, suffix_len = self._encode_suffix(
            trace.events[k:], trace.truncated
        )
        return SuffixItem(
            pair_index=torch.tensor(data=i, dtype=torch.long),
            prefix=self._pad(trace.events[:k]),
            prefix_len=torch.tensor(data=k, dtype=torch.long),
            decoder_input=decoder_input,
            suffix=suffix,
            suffix_len=suffix_len,
            # Read at the last prefix event: what is left to run once the condition has played out.
            remaining_time=torch.tensor(data=trace.remaining_time[k - 1], dtype=torch.float32),
        )

    def pair_info(self, i: int) -> PairInfo:
        """Return which cut of which case the i-th pair is. A generated suffix can be traced back to its case through this"""
        trace, k = self._get_pair(i)
        return PairInfo(case_id=trace.case_id, prefix_len=k, truncated=trace.truncated)

    def length_sorted_indices(self) -> list[int]:
        """Order this split's pairs by how many positions their suffix takes to decode.

        `Decoder.generate` runs a batch until every one of its rows has emitted EOT, so a batch
        of similarly-long suffixes finishes together; unsorted, almost every batch contains one
        early-cut, long-suffix straggler and decodes to nearly the cap regardless of its other
        rows. Sorting the whole split first is what lets that early exit actually save time.

        Returns:
            Pair indices, ascending by suffix length. Passed as a `DataLoader` sampler.
        """
        return sorted(range(len(self)), key=self._suffix_len)

    def _suffix_len(self, i: int) -> int:
        """How many positions the i-th pair's suffix holds: its content, plus an EOT if it has one.

        Cheap to call for every pair of a split, unlike `__getitem__`: it skips the padding
        and tensor work, which sorting has no use for when it only wants a length.
        """
        trace, k = self._get_pair(i)
        content_len = len(trace.events) - k
        return content_len if trace.truncated else content_len + 1

    def _pad(self, events: EncodedSequence) -> EncodedEvents:
        """Copy a run of events into tensors of `max_trace_length`, padding what it leaves over with PAD."""
        length = len(events)

        # Initialize tensors filled with PAD indices for activities and resources, and zeros for the deltas.
        activities = torch.full(
            size=(self.max_len,), fill_value=self.description.activity.pad_index, dtype=torch.long
        )
        resources = torch.full(
            size=(self.max_len,), fill_value=self.description.resource.pad_index, dtype=torch.long
        )
        time_deltas = torch.zeros(size=(self.max_len,), dtype=torch.float32)
        # Zeros for the features too, and not by coincidence: row 0 of the shared table is the
        # PAD every categorical channel uses, which is the point of putting it there.
        feature_categories = torch.zeros(
            size=(self.max_len, self._num_categorical), dtype=torch.long
        )
        feature_values = torch.zeros(size=(self.max_len, self._num_numeric), dtype=torch.float32)
        feature_present = torch.zeros(size=(self.max_len, self._num_numeric), dtype=torch.float32)

        # Copy the actual events into the beginning of the tensors, leaving the rest as PAD/zeros.
        activities[:length] = torch.from_numpy(events.activities)
        resources[:length] = torch.from_numpy(events.resources)
        time_deltas[:length] = torch.from_numpy(events.time_deltas)
        feature_categories[:length] = torch.from_numpy(events.feature_categories)
        feature_values[:length] = torch.from_numpy(events.feature_values)
        feature_present[:length] = torch.from_numpy(events.feature_present)
        return EncodedEvents(
            activities=activities,
            resources=resources,
            time_deltas=time_deltas,
            feature_categories=feature_categories,
            feature_values=feature_values,
            feature_present=feature_present,
        )

    def _encode_suffix(
        self, suffix: EncodedSequence, truncated: bool
    ) -> tuple[torch.Tensor, EncodedEvents, torch.Tensor]:
        """Encode the events from the cut on as the padded suffix (content, EOT, PAD), plus the
        teacher-forced decoder input activities, which are the suffix's activities shifted one
        step right behind a SOS token.

        A suffix cut from a truncated trace gets no EOT: its continuation was cut away, so
        ending it here would teach the model to stop at `max_trace_length` rather than where
        traces really end. The position is left at PAD, which `ignore_index` keeps out of the
        activity/resource loss, and `suffix_len` keeps out of the timestamp loss.

        The feature channels get no EOT token of their own, unlike the resource: EOT is a
        marker rather than an event of the log, the suffix encoder is the only thing that reads
        the position at all, and the two channels that do carry an EOT already identify it.

        Returns:
            The teacher-forced decoder input activities, the padded suffix, and the number of
            real positions in it.
        """
        content_len = len(suffix)
        suffix_len = content_len if truncated else content_len + 1  # EOT closes a complete suffix

        padded_suffix = self._pad(suffix)
        # One position past the content, which the padding left free.
        if not truncated:
            padded_suffix.activities[content_len] = self.description.activity.eot_index
            padded_suffix.resources[content_len] = self.description.resource.eot_index

        # Positions past `suffix_len` are masked out of the loss, so whatever the shift
        # leaves there does not matter. `Decoder` reads no channel of its input but the
        # activity, so that is the only one shifted here.
        decoder_input = _shift_behind(
            padded_suffix.activities,
            first=torch.tensor(data=[self.description.activity.sos_index], dtype=torch.long),
        )

        return decoder_input, padded_suffix, torch.tensor(data=suffix_len, dtype=torch.long)


def _shift_behind(channel: torch.Tensor, *, first: torch.Tensor) -> torch.Tensor:
    """One channel of the teacher-forced decoder input: the padded suffix moved one step later,
    with `first` in the position SOS occupies.

    Args:
        channel: One channel of the padded suffix, `[max_trace_length, ...]`.
        first: What position 0 holds instead, `[1, ...]` matching the channel's trailing axes.
    Returns:
        The shifted channel, the same shape it came in at.
    """
    return torch.cat(tensors=(first, channel[:-1]))


def fixed_subset(dataset: Dataset, *, size: int, generator: torch.Generator) -> Dataset:
    """A random slice of `dataset`, or the whole of it if it is already no bigger.

    Drawn once, so every validation of a run reads the same pairs and two of its points differ
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


def _read_split(
    data_config: DataConfig, *, split: str, description: DatasetDescription
) -> pd.DataFrame:
    """Read one preprocessed split as the values its vocabularies were fit over.

    Each split is a file of its own, so pandas would otherwise infer dtypes for each
    separately and a column of numeric-looking codes could come back as float in one and as
    str in another. Reading every column a vocabulary was fit over as text is what keeps a
    split agreeing with that vocabulary.

    Args:
        data_config: The `data` section, naming the dataset directory.
        split: Which of `train`, `val`, `test` to read.
        description: The dataset description, naming every categorical channel's column.
    Returns:
        The split, one row per event.
    """
    categorical = (description.activity, description.resource, *description.categorical_features)
    text_columns = {CASE_KEY: str} | {column.column: str for column in categorical}
    return read_log(data_config.dir / 'processed' / f'{split}.csv', dtype=text_columns)


def _group_traces(
    split_dataset: pd.DataFrame, max_content_len: int, description: DatasetDescription
) -> list[_Trace]:
    """Group a split into per-case traces, truncated to `max_content_len` events, with their
    content already mapped to indices and normalized floats.

    The split must come from `_read_split`, which is what leaves every column a vocabulary was
    fit over holding the values that vocabulary was fit over.
    """
    # Whole columns at a time, once per split: the same work done per event in `__getitem__`
    # would be repeated for every cut point of every case.
    encoded = encode_events(
        description,
        time_deltas_minutes=split_dataset[EVENT_DELTA_KEY].to_numpy(dtype=np.float32),
        log=split_dataset,
    )
    remaining_time = description.remaining_time.normalize(
        split_dataset[REMAINING_TIME_KEY].to_numpy(dtype=np.float32)
    )

    traces = []

    # One `_Trace` per case, holding that case's slice of the columns encoded above.
    for case_id, group in split_dataset.groupby(CASE_KEY, sort=False):
        # The case's rows as positions into the split-wide columns, cut to the length limit.
        positions = split_dataset.index.get_indexer(group.index)[:max_content_len]
        traces.append(
            _Trace(
                case_id=str(case_id),
                truncated=len(group) > max_content_len,
                events=encoded[positions],
                remaining_time=remaining_time[positions],
            )
        )
    return traces
