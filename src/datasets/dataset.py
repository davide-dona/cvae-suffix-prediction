from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.configs.dataset_info import DatasetInfo
from src.datasets.codec import Codec, EncodedSequence
from src.logs.keys import EVENT_DELTA_KEY, ACTIVITY_KEY, CASE_KEY, RESOURCE_KEY


class PaddedEvents(NamedTuple):
    """One sequence of events, encoded and padded to `max_trace_length`: a prefix or suffix
    cut from a case, or the teacher-forced decoder inputs for a suffix.
    """
    activities: torch.Tensor  # int64, [max_trace_length] per item, [batch_size, max_trace_length] batched
    resources: torch.Tensor
    timestamps: torch.Tensor  # float32 in [0, 1]

    def to(self, device: torch.device) -> "PaddedEvents":
        return PaddedEvents(
            activities=self.activities.to(device),
            resources=self.resources.to(device),
            timestamps=self.timestamps.to(device),
        )


class SuffixItem(NamedTuple):
    """One (prefix, suffix) example, fully encoded: exactly what the DataLoader hands the
    model for one pair, or one batch of pairs.
    """
    pair_index: torch.Tensor      # which pair of the dataset this is; used to trace a prediction back to its case and cut point
    
    prefix: PaddedEvents          # the condition: the events before the cut, no EOT
    prefix_len: torch.Tensor      # real events in `prefix`, the rest being padding
    
    suffix: PaddedEvents          # what the decoder must produce: content, EOT, then padding
    suffix_len: torch.Tensor      # real positions in `suffix`, EOT included where there is one
    
    decoder_input: PaddedEvents   # `suffix` shifted one step behind SOS, for teacher forcing

    def to(self, device: torch.device) -> "SuffixItem":
        """Move a whole batch in one call"""
        return SuffixItem(
            pair_index=self.pair_index.to(device),
            prefix=self.prefix.to(device),
            prefix_len=self.prefix_len.to(device),
            decoder_input=self.decoder_input.to(device),
            suffix=self.suffix.to(device),
            suffix_len=self.suffix_len.to(device),
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


@dataclass(frozen=True)
class PairInfo:
    """The identity of one (prefix, suffix) pair - which case it's from and where it was
    cut - carried separately from its tensors: training never needs it, and writing
    predictions needs nothing else, since a generated suffix means nothing without the case
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
        )

    def pair_info(self, i: int) -> PairInfo:
        """Return which cut of which case the i-th pair is. A prediction can be traced back to its case through this"""
        trace, k = self._get_pair(i)
        return PairInfo(case_id=trace.case_id, prefix_len=k, truncated=trace.truncated)

    def _pad(self, events: EncodedSequence) -> PaddedEvents:
        """Copy a run of events into tensors of `max_trace_length`, padding what it leaves over with PAD."""
        length = len(events)

        # Initialize tensors filled with PAD indices for activities and resources, and zeros for timestamps.
        activities = torch.full(
            size=(self.max_len,), fill_value=self._codec.pad_activity_index, dtype=torch.long
        )
        resources = torch.full(
            size=(self.max_len,), fill_value=self._codec.pad_resource_index, dtype=torch.long
        )
        timestamps = torch.zeros(size=(self.max_len,), dtype=torch.float32)

        # Copy the actual events into the beginning of the tensors, leaving the rest as PAD/zeros.
        activities[:length] = torch.from_numpy(events.activities)
        resources[:length] = torch.from_numpy(events.resources)
        timestamps[:length] = torch.from_numpy(events.timestamps)
        return PaddedEvents(activities=activities, resources=resources, timestamps=timestamps)

    def _encode_suffix(
        self, suffix: EncodedSequence, truncated: bool
    ) -> tuple[PaddedEvents, PaddedEvents, torch.Tensor]:
        """Encode the events from the cut on as the padded suffix (content, EOT, PAD), plus the
        teacher-forced decoder inputs, which are that suffix shifted one step right behind a
        SOS token.

        A suffix cut from a truncated trace gets no EOT: its continuation was cut away, so
        ending it here would teach the model to stop at `max_trace_length` rather than where
        traces really end. The position is left at PAD, which `ignore_index` keeps out of the
        activity/resource loss, and `suffix_len` keeps out of the timestamp loss.

        Returns:
            The teacher-forced decoder inputs, the padded suffix, and the number of real
            positions in it.
        """
        content_len = len(suffix)
        suffix_len = content_len if truncated else content_len + 1  # EOT closes a complete suffix

        padded_suffix = self._pad(suffix)
        # One position past the content, which the padding left free.
        if not truncated:
            padded_suffix.activities[content_len] = self._codec.eot_activity_index
            padded_suffix.resources[content_len] = self._codec.eot_resource_index

        # Positions past `suffix_len` are masked out of the loss, so whatever the shift
        # leaves there does not matter.
        decoder_input = PaddedEvents(
            activities=torch.cat(tensors=(
                torch.tensor(data=[self._codec.sos_activity_index], dtype=torch.long),
                padded_suffix.activities[:-1],
            )),
            resources=torch.cat(tensors=(
                torch.tensor(data=[self._codec.sos_resource_index], dtype=torch.long),
                padded_suffix.resources[:-1],
            )),
            timestamps=torch.cat(
                tensors=(torch.zeros(size=(1,), dtype=torch.float32), padded_suffix.timestamps[:-1])
            ),
        )

        return decoder_input, padded_suffix, torch.tensor(data=suffix_len, dtype=torch.long)


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
                    timestamps=encoded.timestamps[positions],
                ),
            )
        )
    return traces
