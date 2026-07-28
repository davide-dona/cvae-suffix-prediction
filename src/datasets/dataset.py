from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.configs.dataset_info import DatasetInfo
from src.datasets.codec import Codec
from src.logs.keys import EVENT_DELTA_KEY, ACTIVITY_KEY, CASE_KEY, RESOURCE_KEY


@dataclass
class _Trace:
    """
    A single case, encoded once and shared by every (prefix, suffix) pair cut from it.
    The content is truncated to `max_trace_length` events; the EOT token is appended to
    the suffix by `SuffixDataset._encode_suffix`, never stored here.
    """
    truncated: bool  # whether the case was longer than `max_trace_length`, so its real
    # continuation was cut away and no suffix cut from it actually ends. See
    # `SuffixDataset._encode_suffix`.
    activity_idx: np.ndarray  # int64, shape [len]
    resource_idx: np.ndarray  # int64, shape [len(activity_idx)]
    timestamp: np.ndarray  # float32 in [0, 1], shape [len(activity_idx)]


class SuffixDataset(Dataset):
    """A PyTorch Dataset of (prefix, suffix) pairs, cut from every case of one split.

    A case of `n` events yields `n - 1` items, one per cut point `k = 1 .. n - 1`: the
    prefix is `events[:k]` and the suffix is `events[k:]` followed by EOT. Only the
    (case, cut point) index is materialized; the padded tensors are built on demand, so
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
        # One item per cut point. Cases of a single event yield none: there is nothing to
        # condition on before the first event.
        self._pairs: list[tuple[int, int]] = [
            (trace_idx, k)
            for trace_idx, trace in enumerate(self._traces)
            for k in range(1, len(trace.activity_idx))
        ]

    def __len__(self) -> int:
        """Return the number of (prefix, suffix) pairs in this split."""
        return len(self._pairs)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        """Return the i-th (prefix, suffix) pair, padded to `max_trace_length`."""
        trace_idx, k = self._pairs[i]
        trace = self._traces[trace_idx]
        return self._encode_prefix(trace, k) | self._encode_suffix(trace, k)

    def _encode_prefix(self, trace: _Trace, k: int) -> dict[str, torch.Tensor]:
        """Encode `trace`'s first `k` events, right-padded. The prefix carries no EOT: it is
        a cut, not a finished trace."""
        activities = torch.full((self.max_len,), self._codec.pad_activity_index, dtype=torch.long)
        resources = torch.full((self.max_len,), self._codec.pad_resource_index, dtype=torch.long)
        timestamps = torch.zeros(self.max_len, dtype=torch.float32)

        activities[:k] = torch.from_numpy(trace.activity_idx[:k])
        resources[:k] = torch.from_numpy(trace.resource_idx[:k])
        timestamps[:k] = torch.from_numpy(trace.timestamp[:k])

        return {
            'prefix_activities': activities,
            'prefix_resources': resources,
            'prefix_timestamps': timestamps,
            'prefix_len': torch.tensor(k, dtype=torch.long),
        }

    def _encode_suffix(self, trace: _Trace, k: int) -> dict[str, torch.Tensor]:
        """Encode `trace`'s events from `k` on as decoder targets (content, EOT, PAD), plus
        the teacher-forced decoder inputs, which are those targets shifted one step right
        behind a SOS token.

        A truncated trace gets no EOT: its continuation was cut away, so ending its suffix
        here would teach the model to stop at `max_trace_length` rather than where traces
        really end. The position is left at PAD, which `ignore_index` keeps out of the
        activity/resource loss, and `suffix_len` keeps out of the timestamp loss.
        """
        content_len = len(trace.activity_idx) - k
        suffix_len = content_len if trace.truncated else content_len + 1  # EOT closes a complete suffix

        target_activities = torch.full((self.max_len,), self._codec.pad_activity_index, dtype=torch.long)
        target_resources = torch.full((self.max_len,), self._codec.pad_resource_index, dtype=torch.long)
        # EOT/PAD positions are left at 0.0, mirroring how the prefix timestamps are padded.
        target_timestamps = torch.zeros(self.max_len, dtype=torch.float32)

        target_activities[:content_len] = torch.from_numpy(trace.activity_idx[k:])
        target_resources[:content_len] = torch.from_numpy(trace.resource_idx[k:])
        target_timestamps[:content_len] = torch.from_numpy(trace.timestamp[k:])
        if not trace.truncated:
            target_activities[content_len] = self._codec.eot_activity_index
            target_resources[content_len] = self._codec.eot_resource_index

        # Positions past `suffix_len` are masked out of the loss, so whatever the shift
        # leaves there does not matter.
        input_activities = torch.cat((
            torch.tensor([self._codec.sos_activity_index], dtype=torch.long), target_activities[:-1]
        ))
        input_resources = torch.cat((
            torch.tensor([self._codec.sos_resource_index], dtype=torch.long), target_resources[:-1]
        ))
        input_timestamps = torch.cat((torch.zeros(1, dtype=torch.float32), target_timestamps[:-1]))

        return {
            'decoder_input_activities': input_activities,
            'decoder_input_resources': input_resources,
            'decoder_input_timestamps': input_timestamps,
            'target_activities': target_activities,
            'target_resources': target_resources,
            'target_timestamps': target_timestamps,
            'suffix_len': torch.tensor(suffix_len, dtype=torch.long),
        }


def _map_to_index(column: pd.Series, mapping: dict[str, int], *, unk_index: int) -> np.ndarray:
    """Map a whole column of categorical values to vocabulary indices, sending any value the
    train split did not contain to `unk_index`.

    The splits are temporal, so a val/test case can legitimately name a resource that had not
    appeared yet when the vocabulary was fit. UNK is what the model is told about those: not
    which value it was, only that it was none of the ones it was trained on.
    """
    # `map` leaves NaN wherever the value is absent from the vocabulary, which is what UNK covers.
    return column.map(mapping).fillna(unk_index).to_numpy(dtype=np.int64)


def _group_traces(split_dataset: pd.DataFrame, max_content_len: int, codec: Codec) -> list[_Trace]:
    """Group a split into per-case traces, truncated to `max_content_len` events, with their
    content already mapped to indices and normalized floats."""
    # Whole columns at a time, once per split: the same work done per event in `__getitem__`
    # would be repeated for every cut point of every case.
    # `.astype(str)` on both, matching how `DatasetInfo.build` fits the two vocabularies.
    activity_idx_col = _map_to_index(
        split_dataset[ACTIVITY_KEY].astype(str), codec.activity_to_index, unk_index=codec.unk_activity_index
    )
    resource_idx_col = _map_to_index(
        split_dataset[RESOURCE_KEY].astype(str), codec.resource_to_index, unk_index=codec.unk_resource_index
    )
    ts_col = codec.normalize_time(split_dataset[EVENT_DELTA_KEY].to_numpy(dtype=np.float32))

    traces = []

    # One `_Trace` per case, holding that case's slice of the columns encoded above.
    for _, group in split_dataset.groupby(CASE_KEY, sort=False):
        # The case's rows as positions into the split-wide columns, cut to the length limit.
        positions = split_dataset.index.get_indexer(group.index)[:max_content_len]
        traces.append(
            _Trace(
                truncated=len(group) > max_content_len,
                activity_idx=activity_idx_col[positions],
                resource_idx=resource_idx_col[positions],
                timestamp=ts_col[positions],
            )
        )
    return traces
