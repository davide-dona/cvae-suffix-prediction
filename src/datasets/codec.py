from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
import pandas as pd

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


class DecodedSuffix(NamedTuple):
    """One suffix's raw values, decoded from indices and cut to its content length.

    `resources` is empty for a suffix the model generated, which carries no resource channel
    to decode.
    """
    activities: list[str]
    resources: list[str]
    time_deltas_minutes: list[float]


@dataclass(frozen=True)
class EncodedSequence:
    """One run of events, encoded to indices and normalized floats.

    `resources` is None for a sequence the model generated: an event it reads is an activity,
    a resource and a time delta, but an event it writes is an activity and a time delta.
    Anything read out of the log carries all three.
    """
    activities: np.ndarray  # int64, shape [len]
    time_deltas: np.ndarray  # float32 in [0, 1], shape [len(activities)]
    resources: np.ndarray | None = None  # int64, shape [len(activities)]

    def __len__(self) -> int:
        return len(self.activities)

    def __getitem__(self, cut: int | slice | tuple) -> "EncodedSequence":
        """Index every field at once, however numpy would: `[k:]` to cut a trace in two, `[i]`
        or `[i, j]` to pull one sequence out of a batch of them. numpy indexing returns
        views, so this copies nothing.
        """
        return EncodedSequence(
            activities=self.activities[cut],
            time_deltas=self.time_deltas[cut],
            resources=None if self.resources is None else self.resources[cut],
        )


def _map_to_index(column: pd.Series, mapping: dict[str, int], *, unk_index: int) -> np.ndarray:
    """Map a whole column of categorical values to vocabulary indices, sending any value the
    train split did not contain to `unk_index`.

    The splits are temporal, so a val/test case can legitimately name a resource that had not
    appeared yet when the vocabulary was fit. UNK is what the model is told about those: not
    which value it was, only that it was none of the ones it was trained on.
    """
    # `map` leaves NaN wherever the value is absent from the vocabulary, which is what UNK covers.
    return column.map(mapping).fillna(unk_index).to_numpy(dtype=np.int64)


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

    def encode_events(
        self,
        activities: pd.Series,
        resources: pd.Series,
        time_deltas_minutes: np.ndarray,
    ) -> EncodedSequence:
        """Map a run of raw events to the indices and normalized floats the model consumes.

        Args:
            activities: Raw activity values, one row per event.
            resources: Raw resource values, one row per event, aligned with `activities`.
            time_deltas_minutes: Time deltas in minutes, aligned with `activities`.

        Returns:
            The same events as vocabulary indices and a normalized timestamp column.
        """
        return EncodedSequence(
            activities=_map_to_index(activities, self.activity_to_index, unk_index=self.unk_activity_index),
            resources=_map_to_index(resources, self.resource_to_index, unk_index=self.unk_resource_index),
            time_deltas=self.normalize_time(time_deltas_minutes),
        )

    def decode_suffix(self, suffix: EncodedSequence, *, length: int) -> DecodedSuffix:
        """Read one suffix's indices and normalized deltas back to raw values: the inverse of
        `encode_events`, and so taking back the `EncodedSequence` that one returns.

        Args:
            suffix: One suffix as the model holds it, `[seq_len]` per field. A generated one
                has no resource channel, and then neither does what comes back.
            length: How many events to keep; the cut is what drops the EOT a generation ended
                on and the padding behind it, so what comes back holds events and nothing else.

        Returns:
            The suffix's raw activities, resources and time deltas (in minutes), cut to `length`.
        """
        return DecodedSuffix(
            activities=[self.index_to_activity[int(i)] for i in suffix.activities[:length]],
            resources=(
                []
                if suffix.resources is None
                else [self.index_to_resource[int(i)] for i in suffix.resources[:length]]
            ),
            time_deltas_minutes=self.denormalize_time(suffix.time_deltas[:length]).tolist(),
        )

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
