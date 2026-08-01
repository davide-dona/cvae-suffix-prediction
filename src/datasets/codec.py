from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
import pandas as pd

from src.configs.dataset_info import DatasetInfo, TimeStats
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
    to decode. Neither does it carry a time channel: an event the model writes is an
    activity, and the time it predicts is the suffix's remaining time, decoded on its own
    through `Codec.denormalize_remaining_time`.
    """
    activities: list[str]
    resources: list[str]


@dataclass(frozen=True)
class EncodedSequence:
    """One run of events, encoded to indices and normalized floats.

    `resources` and `time_deltas` are both None for a sequence the model generated: an event it
    reads is an activity, a resource and a time delta, but an event it writes is an activity
    alone. Anything read out of the log carries all three.
    """
    activities: np.ndarray  # int64, shape [len]
    time_deltas: np.ndarray | None = None  # float32 in [0, 1], shape [len(activities)]
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
            time_deltas=None if self.time_deltas is None else self.time_deltas[cut],
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

        self._delta_stats = dataset_info.delta_stats
        self._remaining_time_stats = dataset_info.remaining_time_stats

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
            time_deltas=_normalize(time_deltas_minutes, self._delta_stats),
        )

    def decode_suffix(self, suffix: EncodedSequence, *, length: int) -> DecodedSuffix:
        """Read one suffix's indices back to raw values: the inverse of `encode_events`, and so
        taking back the `EncodedSequence` that one returns.

        Args:
            suffix: One suffix as the model holds it, `[seq_len]` per field. A generated one
                has no resource channel, and then neither does what comes back.
            length: How many events to keep; the cut is what drops the EOT a generation ended
                on and the padding behind it, so what comes back holds events and nothing else.

        Returns:
            The suffix's raw activities and resources, cut to `length`.
        """
        return DecodedSuffix(
            activities=[self.index_to_activity[int(i)] for i in suffix.activities[:length]],
            resources=(
                []
                if suffix.resources is None
                else [self.index_to_resource[int(i)] for i in suffix.resources[:length]]
            ),
        )

    def normalize_remaining_time(self, minutes: np.ndarray) -> np.ndarray:
        """Map a case's time left to run into `[0, 1]`, the range the model predicts in."""
        return _normalize(minutes, self._remaining_time_stats)

    def denormalize_remaining_time(self, normalized: np.ndarray) -> np.ndarray:
        """Read the model's remaining-time prediction back as minutes, which is what the
        evaluation reports and the log can be compared against.

        Approximate at the top of the range: clipping is lossy, so anything that was above
        `clip_value` on the way in comes back as `clip_value`.
        """
        stats = self._remaining_time_stats
        span = stats.log_max - stats.log_min
        return np.expm1(np.asarray(normalized, dtype=np.float64) * span + stats.log_min)


def _normalize(minutes: np.ndarray, stats: TimeStats) -> np.ndarray:
    """log1p + min-max into `[0, 1]`, clipping to a range fit on the train split.

    Clipping bounds the outliers and log1p then pulls in the long tail, so the bulk of the
    values do not all end up squeezed into the bottom of the range: both columns here span
    several orders of magnitude.

    Args:
        minutes: The raw values to normalize.
        stats: The range to normalize into, fit on train by `DatasetInfo`.
    Returns:
        The same values in `[0, 1]`, as float32.
    """
    log_minutes = np.log1p(np.clip(minutes, 0.0, stats.clip_value))
    span = stats.log_max - stats.log_min
    # A split where every value is identical would divide by zero here.
    if span <= 0:
        return np.zeros_like(log_minutes, dtype=np.float32)
    return np.clip((log_minutes - stats.log_min) / span, 0.0, 1.0).astype(np.float32)
