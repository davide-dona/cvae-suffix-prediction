from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.datasets.description import CategoricalColumn, DatasetDescription

@dataclass(frozen=True)
class EncodedSequence:
    """One run of events, encoded to indices and normalized floats."""
    activities: np.ndarray          # int64, shape [len]
    time_deltas: np.ndarray         # float32 in [0, 1], shape [len(activities)]
    resources: np.ndarray           # int64, shape [len(activities)]
    feature_categories: np.ndarray  # int64, shape [len(activities), num_categorical]
    feature_values: np.ndarray      # float32 in [0, 1], shape [len(activities), num_numeric]
    feature_present: np.ndarray     # float32 0/1, shape [len(activities), num_numeric]

    def __len__(self) -> int:
        return len(self.activities)

    def __getitem__(self, cut: int | slice | np.ndarray) -> "EncodedSequence":
        """Return a cut of the sequence - a slice or an index selection - with all fields aligned."""
        return EncodedSequence(
            activities=self.activities[cut],
            time_deltas=self.time_deltas[cut],
            resources=self.resources[cut],
            feature_categories=self.feature_categories[cut],
            feature_values=self.feature_values[cut],
            feature_present=self.feature_present[cut],
        )


def encode_events(
    description: DatasetDescription,
    *,
    time_deltas_minutes: np.ndarray,
    log: pd.DataFrame,
) -> EncodedSequence:
    """Map a run of raw events to the indices and normalized floats the model consumes.

    Args:
        description: The dataset's description, naming every channel read here and holding the
            vocabulary or range each is encoded through.
        time_deltas_minutes: Time deltas in minutes, one per row of `log`.
        log: The events as a preprocessed split holds them. The whole frame, not a selection of
            it: which columns each channel reads is the description's answer, and this is where
            it is asked.

    Returns:
        The same events as vocabulary indices and normalized columns.
    """
    feature_values, feature_present = _encode_numerics(description, log)
    return EncodedSequence(
        activities=_encode_column(description.activity, log),
        resources=_encode_column(description.resource, log),
        time_deltas=description.delta.normalize(time_deltas_minutes),
        feature_categories=_encode_categories(description, log),
        feature_values=feature_values,
        feature_present=feature_present,
    )


@dataclass(frozen=True)
class DecodedSequence:
    """One run of events, decoded back to the log's own units. A channel the model learns to
    write gains a field here."""
    activities: list[str]
    remaining_time_minutes: float

    def __len__(self) -> int:
        return len(self.activities)


def decode_activities(
    description: DatasetDescription,
    *,
    activities: np.ndarray,
    length: int,
) -> list[str]:
    """Read a run of activity indices back into the log's own names.

    Args:
        description: The dataset's description, holding the activity channel's decode map.
        activities: The activity indices of one sequence, int64, `[seq_len]`.
        length: How many of them to keep, the rest being padding.
    Returns:
        The activity names, in order.
    """
    return [description.activity.from_index[int(index)] for index in activities[:length]]


def decode_sequence(
    description: DatasetDescription,
    *,
    activities: np.ndarray,
    length: int,
    remaining_time: float,
) -> DecodedSequence:
    """Read one sequence back into the log's own units, the inverse of `encode_events` over
    every channel the model writes.

    Args:
        description: The dataset's description, holding each written channel's decode map or
            normalization range.
        activities: The activity indices of one sequence, int64, `[seq_len]`.
        length: How many events to keep; the cut is what drops the EOT a generation ended
            on and the padding behind it, so what comes back holds events and nothing else.
        remaining_time: The sequence's remaining time as the model writes it, normalized in
            `[0, 1]`.

    Returns:
        The sequence in raw activity names and minutes.
    """
    return DecodedSequence(
        activities=decode_activities(description, activities=activities, length=length),
        remaining_time_minutes=float(description.remaining_time.denormalize(remaining_time)),
    )


def _encode_column(column: CategoricalColumn, log: pd.DataFrame) -> np.ndarray:
    """One categorical channel as rows of the table it is embedded with, with an UNK for values
    the train split did not see.

    Returns:
        `[len(log)]` of int64 rows.
    """
    return (
        log[column.column]
        .map(column.to_index)
        .fillna(column.unk_index)
        .to_numpy(dtype=np.int64)
    )


def _encode_categories(description: DatasetDescription, log: pd.DataFrame) -> np.ndarray:
    """Every categorical feature channel of a run of events, packed into one index array.

    Args:
        description: The dataset's description, holding the feature channels and their blocks
            of the shared table.
        log: The events as a preprocessed split holds them, so a gap in a channel already carries
            the missing token preprocessing gave it.
    Returns:
        `[len(log), num_categorical]` of rows of the shared embedding table.
    """
    if not description.categorical_features:
        return np.zeros(shape=(len(log), 0), dtype=np.int64)
    return np.stack(
        arrays=[_encode_column(feature, log) for feature in description.categorical_features],
        axis=1,
    )


def _encode_numerics(
    description: DatasetDescription, log: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Every numeric feature channel's normalized value, and the flag saying it was there.

    Args:
        description: The dataset's description, holding the feature channels and their ranges.
        log: The events as rows of the log.
    Returns:
        The values and the flags, `[len(log), num_numeric]` each. A missing value is 0.0 with
        a 0.0 flag; 0.0 is also a legitimate normalized value, which is what the flag is for.
    """
    if not description.numeric_features:
        empty = np.zeros(shape=(len(log), 0), dtype=np.float32)
        return empty, empty.copy()

    values, present = [], []
    for feature in description.numeric_features:
        raw = log[feature.column].to_numpy(dtype=np.float64)
        flags = np.isfinite(raw).astype(np.float32)
        # The range is fit on the finite values, so the missing ones are zeroed rather than
        # normalized: multiplying by the flag is what pins them to exactly 0.0.
        finite = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        values.append(feature.normalize(finite) * flags)
        present.append(flags)
    return np.stack(arrays=values, axis=1), np.stack(arrays=present, axis=1)
