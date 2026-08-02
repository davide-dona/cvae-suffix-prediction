from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

from src.configs.dataset_info import DatasetInfo
from src.datasets.codec import Codec, EncodedSequence
from src.logs.keys import EVENT_DELTA_KEY, ACTIVITY_KEY, CASE_KEY, REMAINING_TIME_KEY, RESOURCE_KEY


class EncodedEvents(NamedTuple):
    """One sequence of events as the model reads it: vocabulary indices and normalized floats,
    padded to `max_trace_length`. A prefix or suffix cut from a case.

    The same fields as `codec.EncodedSequence`, one stage on: those are numpy arrays of one
    trace's real length, these are the padded tensors a batch is collated from.
    """
    activities: torch.Tensor  # int64, [max_trace_length] per item, [batch_size, max_trace_length] batched
    resources: torch.Tensor
    time_deltas: torch.Tensor  # float32 in [0, 1]
    # The log's own columns, one axis wide rather than one field per column. Zero-width on a
    # dataset whose config names none.
    feature_categories: torch.Tensor  # int64, [max_trace_length, num_categorical] per item
    feature_values: torch.Tensor      # float32 in [0, 1], [max_trace_length, num_numeric] per item
    feature_present: torch.Tensor     # float32 0/1, 0.0 where the log had no value

    def to(self, device: torch.device) -> "EncodedEvents":
        return EncodedEvents(
            activities=self.activities.to(device),
            resources=self.resources.to(device),
            time_deltas=self.time_deltas.to(device),
            feature_categories=self.feature_categories.to(device),
            feature_values=self.feature_values.to(device),
            feature_present=self.feature_present.to(device),
        )

class SuffixItem(NamedTuple):
    """One (prefix, suffix) example, fully encoded: exactly what the DataLoader hands the
    model for one pair, or one batch of pairs.
    """
    pair_index: torch.Tensor      # which pair of the dataset this is; used to trace a prediction back to its case and cut point

    prefix: EncodedEvents          # the condition: the events before the cut, no EOT
    prefix_len: torch.Tensor      # real events in `prefix`, the rest being padding

    suffix: EncodedEvents          # what the decoder must produce: content, EOT, then padding
    suffix_len: torch.Tensor      # real positions in `suffix`, EOT included where there is one

    # `suffix` activities shifted one step behind SOS, for teacher forcing. int64,
    # [max_trace_length] per item, [batch_size, max_trace_length] batched. Only the
    # activity channel is needed: `Decoder` reads no other channel of its input.
    decoder_input: torch.Tensor

    # The other half of what the model predicts: minutes from the last prefix event to the end
    # of the case, normalized into [0, 1]. One scalar for the whole pair, measured against the
    # case's real ending even where the suffix above was truncated.
    remaining_time: torch.Tensor  # float32, [] per item, [batch_size] batched

    def to(self, device: torch.device) -> "SuffixItem":
        """Move a whole batch in one call"""
        return SuffixItem(
            pair_index=self.pair_index.to(device),
            prefix=self.prefix.to(device),
            prefix_len=self.prefix_len.to(device),
            decoder_input=self.decoder_input.to(device),
            suffix=self.suffix.to(device),
            suffix_len=self.suffix_len.to(device),
            remaining_time=self.remaining_time.to(device),
        )


@dataclass(frozen=True)
class _Trace:
    """One case of the log (one process instance), encoded once and shared by every
    (prefix, suffix) pair cut from it. `truncated` marks a case longer than
    `max_trace_length`: none of its cuts end in a real EOT (see `SuffixDataset._encode_suffix`).
    """
    case_id: str  # which case of the log this is, kept for `SuffixDataset.pair_info`
    truncated: bool  # whether the case was longer than `max_trace_length`, so its real
    # continuation was cut away and no suffix cut from it actually ends. See
    # `SuffixDataset._encode_suffix`.
    events: EncodedSequence
    # Normalized minutes from each event to the case's real ending, one per event of
    # `events`. Cutting after `k` events leaves `remaining_time[k - 1]` still to run, so a
    # truncated case still carries the true value at every event that survived the cut.
    remaining_time: np.ndarray  # float32 in [0, 1], shape [len(events)]


@dataclass(frozen=True)
class PairInfo:
    """The identity of one (prefix, suffix) pair - which case it's from and where it was
    cut - carried separately from its tensors: training never needs it, and writing
    generations needs nothing else, since a generated suffix means nothing without the case
    and cut point it continues.
    """
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

    def __init__(self, split_dataset: pd.DataFrame, dataset_info: DatasetInfo, codec: Codec):
        self._codec = codec
        self.max_len = dataset_info.max_trace_length
        self._num_categorical = len(dataset_info.categorical_features)
        self._num_numeric = len(dataset_info.numeric_features)

        # Group the split into per-case traces, truncated to max_len events, with their content
        # mapped to indices once here rather than on every __getitem__ call.
        self._traces = _group_traces(split_dataset, self.max_len, codec)
        
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

    def _pad(self, events: EncodedSequence) -> EncodedEvents:
        """Copy a run of events into tensors of `max_trace_length`, padding what it leaves over with PAD."""
        length = len(events)

        # Initialize tensors filled with PAD indices for activities and resources, and zeros for the deltas.
        activities = torch.full(
            size=(self.max_len,), fill_value=self._codec.pad_activity_index, dtype=torch.long
        )
        resources = torch.full(
            size=(self.max_len,), fill_value=self._codec.pad_resource_index, dtype=torch.long
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
            padded_suffix.activities[content_len] = self._codec.eot_activity_index
            padded_suffix.resources[content_len] = self._codec.eot_resource_index

        # Positions past `suffix_len` are masked out of the loss, so whatever the shift
        # leaves there does not matter. `Decoder` reads no channel of its input but the
        # activity, so that is the only one shifted here.
        decoder_input = _shift_behind(
            padded_suffix.activities,
            first=torch.tensor(data=[self._codec.sos_activity_index], dtype=torch.long),
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


def _group_traces(split_dataset: pd.DataFrame, max_content_len: int, codec: Codec) -> list[_Trace]:
    """Group a split into per-case traces, truncated to `max_content_len` events, with their
    content already mapped to indices and normalized floats."""
    # Whole columns at a time, once per split: the same work done per event in `__getitem__`
    # would be repeated for every cut point of every case.
    # `.astype(str)` on both, matching how `DatasetInfo.build` fits the two vocabularies.
    encoded = codec.encode_events(
        activities=split_dataset[ACTIVITY_KEY].astype(str),
        resources=split_dataset[RESOURCE_KEY].astype(str),
        time_deltas_minutes=split_dataset[EVENT_DELTA_KEY].to_numpy(dtype=np.float32),
        log=split_dataset,
    )
    remaining_time = codec.normalize_remaining_time(
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
                events=EncodedSequence(
                    activities=encoded.activities[positions],
                    resources=encoded.resources[positions],
                    time_deltas=encoded.time_deltas[positions],
                    feature_categories=encoded.feature_categories[positions],
                    feature_values=encoded.feature_values[positions],
                    feature_present=encoded.feature_present[positions],
                ),
                remaining_time=remaining_time[positions],
            )
        )
    return traces
