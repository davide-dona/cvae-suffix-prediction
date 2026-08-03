from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
import pandas as pd

from src.datasets.info import DatasetInfo, NormalizationRange, as_categories
from src.logs.keys import (
    EOT_ACTIVITY,
    EOT_RESOURCE,
    PADDING_ACTIVITY,
    PADDING_RESOURCE,
    SOS_ACTIVITY,
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

    Every field but `activities` is None for a sequence the model generated: an event it reads
    is an activity, a resource, a time delta and whatever feature channels the log offers, but
    an event it writes is an activity alone. Anything read out of the log carries all of them.
    """
    activities: np.ndarray  # int64, shape [len]
    time_deltas: np.ndarray | None = None  # float32 in [0, 1], shape [len(activities)]
    resources: np.ndarray | None = None  # int64, shape [len(activities)]
    # The log's own columns, packed one axis wide rather than one field per column: a log with 25
    # of them would otherwise carry 25 arrays through every cut, pad and collate.
    feature_categories: np.ndarray | None = None  # int64, shape [len(activities), num_categorical]
    feature_values: np.ndarray | None = None      # float32 in [0, 1], shape [len(activities), num_numeric]
    feature_present: np.ndarray | None = None     # float32 0/1, shape [len(activities), num_numeric]

    def __len__(self) -> int:
        return len(self.activities)

    def __getitem__(self, cut: int | slice | tuple) -> "EncodedSequence":
        """Index every field at once, however numpy would: `[k:]` to cut a trace in two, `[i]`
        or `[i, j]` to pull one sequence out of a batch of them. numpy indexing returns
        views, so this copies nothing.

        `cut` addresses the event axes only. The feature fields carry one trailing feature axis
        that every such index passes through untouched; an index naming as many axes as
        `activities` has would reach into it instead.
        """
        return EncodedSequence(
            activities=self.activities[cut],
            time_deltas=None if self.time_deltas is None else self.time_deltas[cut],
            resources=None if self.resources is None else self.resources[cut],
            feature_categories=(
                None if self.feature_categories is None else self.feature_categories[cut]
            ),
            feature_values=None if self.feature_values is None else self.feature_values[cut],
            feature_present=(
                None if self.feature_present is None else self.feature_present[cut]
            ),
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
        self.unk_resource_index = dataset_info.unk_resource_index
        self.index_to_resource = {i: r for r, i in self.resource_to_index.items()} | {
            self.eot_resource_index: EOT_RESOURCE,
            self.pad_resource_index: PADDING_RESOURCE,
            self.unk_resource_index: UNK_RESOURCE,
        }

        self._delta_stats = dataset_info.delta_stats
        self._remaining_time_stats = dataset_info.remaining_time_stats

        self._categorical_features = dataset_info.categorical_features
        self._numeric_features = dataset_info.numeric_features
        # The table offsets are folded into the maps here rather than added on every lookup, so
        # what `encode_events` returns is already a row of the table the model embeds with.
        self._category_to_index = tuple(
            {value: feature.offset + i for i, value in enumerate(feature.vocab)}
            for feature in self._categorical_features
        )

    def encode_events(
        self,
        activities: pd.Series,
        resources: pd.Series,
        time_deltas_minutes: np.ndarray,
        log: pd.DataFrame,
    ) -> EncodedSequence:
        """Map a run of raw events to the indices and normalized floats the model consumes.

        Args:
            activities: Raw activity values, one row per event.
            resources: Raw resource values, one row per event, aligned with `activities`.
            time_deltas_minutes: Time deltas in minutes, aligned with `activities`.
            log: The same events as rows of the log, from which the event features this codec was
                configured with are read. The whole frame, not a selection of it: which columns
                those are is `DatasetInfo`'s answer, and this is where it is asked.

        Returns:
            The same events as vocabulary indices and normalized columns.
        """
        feature_values, feature_present = self._encode_numerics(log)
        return EncodedSequence(
            activities=_map_to_index(activities, self.activity_to_index, unk_index=self.unk_activity_index),
            resources=_map_to_index(resources, self.resource_to_index, unk_index=self.unk_resource_index),
            time_deltas=_normalize(time_deltas_minutes, self._delta_stats),
            feature_categories=self._encode_categories(log),
            feature_values=feature_values,
            feature_present=feature_present,
        )

    def _encode_categories(self, log: pd.DataFrame) -> np.ndarray:
        """Every categorical feature channel of a run of events, packed into one index array.

        Args:
            log: The events as rows of the log.
        Returns:
            `[len(log), num_categorical]` of rows of the shared embedding table.
        """
        if not self._categorical_features:
            return np.zeros(shape=(len(log), 0), dtype=np.int64)
        return np.stack(
            arrays=[
                _map_to_index(
                    as_categories(log[feature.column]), mapping, unk_index=feature.unk_index
                )
                for feature, mapping in zip(self._categorical_features, self._category_to_index)
            ],
            axis=1,
        )

    def _encode_numerics(self, log: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Every numeric feature channel's normalized value, and the flag saying it was there.

        Args:
            log: The events as rows of the log.
        Returns:
            The values and the flags, `[len(log), num_numeric]` each. A missing value is 0.0 with
            a 0.0 flag; 0.0 is also a legitimate normalized value, which is what the flag is for.
        """
        if not self._numeric_features:
            empty = np.zeros(shape=(len(log), 0), dtype=np.float32)
            return empty, empty.copy()

        values, present = [], []
        for feature in self._numeric_features:
            raw = log[feature.column].to_numpy(dtype=np.float64)
            flags = np.isfinite(raw).astype(np.float32)
            # The range is fit on the finite values, so the missing ones are zeroed rather than
            # normalized: multiplying by the flag is what pins them to exactly 0.0.
            values.append(_normalize(np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), feature.range) * flags)
            present.append(flags)
        return np.stack(arrays=values, axis=1), np.stack(arrays=present, axis=1)

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


def _normalize(values: np.ndarray, stats: NormalizationRange) -> np.ndarray:
    """log1p + min-max into `[0, 1]`, clipping to a range fit on the train split.

    Clipping bounds the outliers and log1p then pulls in the long tail, so the bulk of the
    values do not all end up squeezed into the bottom of the range: the time columns and the
    amounts beside them span several orders of magnitude.

    Args:
        values: The raw values to normalize.
        stats: The range to normalize into, fit on train by `DatasetInfo`.
    Returns:
        The same values in `[0, 1]`, as float32.
    """
    log_values = np.log1p(np.clip(values, 0.0, stats.clip_value))
    span = stats.log_max - stats.log_min
    # A split where every value is identical would divide by zero here.
    if span <= 0:
        return np.zeros_like(log_values, dtype=np.float32)
    return np.clip((log_values - stats.log_min) / span, 0.0, 1.0).astype(np.float32)
