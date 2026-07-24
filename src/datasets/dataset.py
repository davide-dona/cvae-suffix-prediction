from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.configs.dataset_info import DatasetInfo
from src.datasets.encoding import Encoding
from src.logs.keys import EVENT_DELTA_KEY, ACTIVITY_KEY, CASE_KEY, RESOURCE_KEY


@dataclass
class _Trace:
    """
    A single trace, with its content encoded as indices and its condition features in raw form.
    The content is truncated to `max_trace_len - 1` events, leaving room for the EOT token to be added later by `GenericDataset._encode_x`.
    """
    activity_idx: np.ndarray  # int64, shape [len]
    resource_idx: np.ndarray  # int64, shape [len(activity_idx)]
    timestamp: np.ndarray  # float32 in [0, 1], shape [len(activity_idx)]
    attributes: dict[str, np.ndarray]  # float32, shape [len(activity_idx)] each: vocab index
    # (categorical) or min-max normalized value (numeric), see `Encoding.encode_attribute_column`
    condition: dict[str, object]  # raw (pre-encoding) value per condition feature


class GenericDataset(Dataset):
    """A PyTorch Dataset for a single split of a log, with traces grouped and encoded."""
    def __init__(self, split_dataset: pd.DataFrame, dataset_info: DatasetInfo, encoding: Encoding):
        self._dataset_info = dataset_info
        self._encoding = encoding
        self.max_len = dataset_info.max_trace_length

        # Leave space for the EOT token, which is added by _encode_x
        max_content_len = self.max_len - 1
        # Group the split into per-case traces, truncated to max_content_len events, and encode the content as indices
        self._traces = _group_traces(
            split_dataset, dataset_info.condition_features, dataset_info.attribute_features, max_content_len, encoding
        )

    def __len__(self) -> int:
        """Return the number of traces in this split."""
        return len(self._traces)

    def __getitem__(self, i: int):
        """Return the i-th trace, encoded as a reconstruction target (content, EOT, PAD) and its condition features."""
        # Encode the i-th trace as a reconstruction target (content, EOT, PAD)
        trace = self._traces[i]
        x = self._encode_x(trace)
        # Encode the condition features as a tensor of indices
        y = torch.from_numpy(self._encoding.encode_condition(trace.condition))
        return x, y

    def _encode_x(self, trace: _Trace) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the whole trace as a reconstruction target: content, then EOT, then PAD"""
        # Get the number of events in the trace (before adding EOT and PAD)
        n = len(trace.activity_idx)

        # Initialize the output tensors with PAD values
        act_idx = torch.full((self.max_len,), self._encoding.pad_activity_index, dtype=torch.long)
        res_idx = torch.full((self.max_len,), self._encoding.pad_resource_index, dtype=torch.long)
        ts = torch.zeros(self.max_len, dtype=torch.float32)
        # EOT/PAD positions are left at 0.0, mirroring how `ts` is padded above.
        attributes = {name: torch.zeros(self.max_len, dtype=torch.float32) for name in trace.attributes}

        # Fill in the content part of the output tensors with the trace's encoded content
        if n > 0:
            act_idx[:n] = torch.from_numpy(trace.activity_idx)
            res_idx[:n] = torch.from_numpy(trace.resource_idx)
            ts[:n] = torch.from_numpy(trace.timestamp)
            for name, values in trace.attributes.items():
                attributes[name][:n] = torch.from_numpy(values)

        assert n < self.max_len, (
            "content left no room for the EOT token; traces should already be truncated "
            "to max_seq_len - 1 events by _group_traces"
        )
        # Add the EOT token at the end of the content
        act_idx[n] = self._encoding.eot_activity_index
        res_idx[n] = self._encoding.eot_resource_index
        return attributes, act_idx, ts, res_idx


def _map_to_index(column: pd.Series, mapping: dict[str, int], *, kind: str) -> np.ndarray:
    """Map a column of categorical values to indices using the provided mapping, raising an error for unseen values."""
    # Map the column to indices using the provided mapping
    encoded = column.map(mapping)
    
    # Check for unseen values (NaN) in the encoded column
    unseen = encoded.isna()
    if unseen.any():
        raise KeyError(
            f"{kind} value(s) {sorted(set(column[unseen]))!r} were not seen in the train "
            "split; Encoding is fit train-only and does not support unseen values at val/test time"
        )
    
    return encoded.to_numpy(dtype=np.int64)


def _group_traces(
    split_dataset: pd.DataFrame,
    condition_features: list[str],
    attribute_features: list[str],
    max_content_len: int,
    encoding: Encoding,
) -> list[_Trace]:
    """Group a split into per-case traces, truncated to max_content_len events, and encode the content as indices"""
    # Map the activity and resource columns to indices using the provided encoding, and normalize the timestamps
    activity_idx_col = _map_to_index(split_dataset[ACTIVITY_KEY], encoding.activity_to_index, kind="activity")
    resource_idx_col = _map_to_index(split_dataset[RESOURCE_KEY].astype(str), encoding.resource_to_index, kind="resource")
    ts_col = encoding.normalize_time(split_dataset[EVENT_DELTA_KEY].to_numpy(dtype=np.float32))
    # Encode each attribute column once, up front, same reasoning as activity/resource/ts above
    attribute_cols = {
        name: encoding.encode_attribute_column(name, split_dataset[name]) for name in attribute_features
    }

    traces = []

    # Group the split dataset by case, and for each case, create a _Trace object with the encoded content and condition features
    for _, group in split_dataset.groupby(CASE_KEY, sort=False):
        condition = {feature: group[feature].iloc[0] for feature in condition_features}
        positions = split_dataset.index.get_indexer(group.index)[:max_content_len]
        traces.append(
            _Trace(
                activity_idx=activity_idx_col[positions],
                resource_idx=resource_idx_col[positions],
                timestamp=ts_col[positions],
                attributes={name: col[positions] for name, col in attribute_cols.items()},
                condition=condition,
            )
        )
    return traces
