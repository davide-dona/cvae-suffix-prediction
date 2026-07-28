from __future__ import annotations
import numpy as np

from src.configs.dataset_info import DatasetInfo
from src.logs.keys import (
    EOT_ACTIVITY,
    EOT_RESOURCE,
    PADDING_ACTIVITY,
    PADDING_RESOURCE,
    SOS_ACTIVITY,
    SOS_RESOURCE,
    UNK_ACTIVITY,
    UNK_RESOURCE,
)


class Codec:
    """Two-way mapping between a dataset's raw values and the indices and normalized floats
    the model consumes.

    Nothing here is learned: it is lookup tables and a normalization range, all fit on the
    train split and then applied unchanged to val/test. That is the distinction from the
    model's `TraceEncoder`, which is what turns these indices into representations.

    The special-token indices are taken from `DatasetInfo`, which owns the vocabulary layout,
    so the embeddings a model allocates and the indices this produces cannot drift apart.
    """

    def __init__(self, dataset_info: DatasetInfo):
        self.activity_to_index = {activity: i for i, activity in enumerate(dataset_info.activity_vocab)}
        self.eot_activity_index = dataset_info.eot_activity_index
        self.pad_activity_index = dataset_info.pad_activity_index
        self.sos_activity_index = dataset_info.sos_activity_index
        self.unk_activity_index = dataset_info.unk_activity_index
        # The decode direction, for reading a prediction back as a trace. The special tokens are
        # added here only: no raw value maps to them, so they have no entry going the other way.
        # UNK is the exception in spirit - many raw values map to it - but it is still one-way,
        # since which of them it was is exactly what the encoding threw away.
        self.index_to_activity = {i: a for a, i in self.activity_to_index.items()} | {
            self.eot_activity_index: EOT_ACTIVITY,
            self.pad_activity_index: PADDING_ACTIVITY,
            self.sos_activity_index: SOS_ACTIVITY,
            self.unk_activity_index: UNK_ACTIVITY,
        }

        self.resource_to_index = {resource: i for i, resource in enumerate(dataset_info.resource_vocab)}
        self.eot_resource_index = dataset_info.eot_resource_index
        self.pad_resource_index = dataset_info.pad_resource_index
        self.sos_resource_index = dataset_info.sos_resource_index
        self.unk_resource_index = dataset_info.unk_resource_index
        self.index_to_resource = {i: r for r, i in self.resource_to_index.items()} | {
            self.eot_resource_index: EOT_RESOURCE,
            self.pad_resource_index: PADDING_RESOURCE,
            self.sos_resource_index: SOS_RESOURCE,
            self.unk_resource_index: UNK_RESOURCE,
        }

        self._time_stats = dataset_info.time_stats

    def normalize_time(self, delta_minutes: np.ndarray) -> np.ndarray:
        """log1p + min-max into `[0, 1]`, clipping to the range fit on the train split."""
        stats = self._time_stats
        # Clipping bounds the outliers, and log1p then pulls in the long tail, so the bulk of
        # the deltas do not all end up squeezed into the bottom of the range.
        clipped = np.clip(delta_minutes, 0.0, stats.clip_value)
        log_delta = np.log1p(clipped)
        span = stats.log_max - stats.log_min
        # A split where every delta is identical would divide by zero here.
        if span <= 0:
            return np.zeros_like(log_delta, dtype=np.float32)
        return np.clip((log_delta - stats.log_min) / span, 0.0, 1.0).astype(np.float32)

    def denormalize_time(self, normalized: np.ndarray) -> np.ndarray:
        """Invert `normalize_time`, back to minutes. Approximate for deltas that were clipped
        on the way in: clipping is lossy, so anything above `clip_value` comes back as
        `clip_value`.
        """
        stats = self._time_stats
        span = stats.log_max - stats.log_min
        log_delta = np.asarray(normalized, dtype=np.float64) * span + stats.log_min
        return np.expm1(log_delta)
