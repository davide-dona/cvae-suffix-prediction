from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
import torch

from src.datasets.description import DatasetDescription


class Events(NamedTuple):
    """The set of events composing a single trace, as the model reads it."""

    activities: torch.Tensor  # int64, [..., seq_len]
    resources: torch.Tensor  # int64, [..., seq_len]
    time_deltas: torch.Tensor  # float32 in [0, 1], [..., seq_len]
    feature_categories: torch.Tensor  # int64, [..., seq_len, num_categorical]
    feature_values: torch.Tensor  # float32 in [0, 1], [..., seq_len, num_numeric]
    feature_present: (
        torch.Tensor
    )  # float32 0/1, 0.0 where the log had no value [..., seq_len, num_numeric]
    length: torch.Tensor  # int64, the real, unpadded length of the run, [...], 1D

    def cut(self, index: slice | torch.Tensor) -> Events:
        """Slice every per-event field by index, updating `length`."""
        activities = self.activities[index]
        return Events(
            activities=activities,
            resources=self.resources[index],
            time_deltas=self.time_deltas[index],
            feature_categories=self.feature_categories[index],
            feature_values=self.feature_values[index],
            feature_present=self.feature_present[index],
            length=torch.tensor(data=len(activities), dtype=torch.long),
        )

    def padded(self, to: int) -> Events:
        """Pad every per-event field to `to` positions, leaving `length` unchanged.
        This allows a batch of runs to be stacked into a single tensor, with the padding masked
        out by `pad_mask`.

        Args:
            to: The width to pad to, `max_trace_length` for everything the model reads.
        Returns:
            The same events at the front of `to` positions, `length` unchanged.
        """
        # Every field but `length`, which counts real events and so survives the padding.
        return Events(
            *(
                torch.cat(
                    tensors=(
                        channel,
                        torch.zeros(
                            size=(to - channel.size(dim=0), *channel.shape[1:]), dtype=channel.dtype
                        ),
                    )
                )
                for channel in self[:-1]
            ),
            length=self.length,
        )

    def pad_mask(self) -> torch.Tensor:
        """Marks the positions that were padded out.
        Returns:
            `[batch_size, seq_len]`, True where the position holds padding.
        """
        positions = torch.arange(end=self.activities.size(dim=-1), device=self.length.device)
        return positions.unsqueeze(dim=0) >= self.length.unsqueeze(dim=1)

    def to(self, device: torch.device) -> Events:
        """Move a whole batch in one call"""
        return Events(*(field.to(device) for field in self))


def encode_events(
    description: DatasetDescription,
    *,
    time_deltas_minutes: np.ndarray,
    log: pd.DataFrame,
) -> Events:
    """Map a run of raw events to the indices and normalized floats the model consumes.

    Args:
        description: The dataset's description, naming every channel read here and holding the
            vocabulary or range each is encoded through.
        time_deltas_minutes: Time deltas in minutes, one per row of `log`.
        log: The events as a preprocessed split holds them. The whole frame, not a selection of
            it: which columns each channel reads is the description's answer, and this is where
            it is asked.

    Returns:
        The same events as vocabulary indices and normalized channels, unpadded, so every one of
        them counts towards `length`.
    """
    feature_values, feature_present = _encode_numerics(description, log)
    return Events(
        # `torch.tensor` rather than `from_numpy`: pandas hands back a read-only view of its
        # own block for some dtypes, which torch would wrap rather than copy.
        activities=torch.tensor(data=description.activity.encode(log), dtype=torch.long),
        resources=torch.tensor(data=description.resource.encode(log), dtype=torch.long),
        time_deltas=torch.from_numpy(description.delta.normalize(time_deltas_minutes)),
        feature_categories=_encode_categories(description, log),
        feature_values=feature_values,
        feature_present=feature_present,
        length=torch.tensor(data=len(log), dtype=torch.long),
    )


def _encode_categories(description: DatasetDescription, log: pd.DataFrame) -> torch.Tensor:
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
        return torch.zeros(size=(len(log), 0), dtype=torch.long)
    return torch.stack(
        tensors=[
            torch.tensor(data=feature.encode(log), dtype=torch.long)
            for feature in description.categorical_features
        ],
        dim=1,
    )


def _encode_numerics(
    description: DatasetDescription, log: pd.DataFrame
) -> tuple[torch.Tensor, torch.Tensor]:
    """Every numeric feature channel's normalized value, and the flag saying it was there.

    Args:
        description: The dataset's description, holding the feature channels and their ranges.
        log: The events as rows of the log.
    Returns:
        The values and the flags, `[len(log), num_numeric]` each. A missing value is 0.0 with
        a 0.0 flag; 0.0 is also a legitimate normalized value, which is what the flag is for.
    """
    if not description.numeric_features:
        empty = torch.zeros(size=(len(log), 0), dtype=torch.float32)
        return empty, empty.clone()

    values, present = [], []
    for feature in description.numeric_features:
        raw = log[feature.column].to_numpy(dtype=np.float64)
        flags = np.isfinite(raw).astype(np.float32)
        # The range is fit on the finite values, so the missing ones are zeroed rather than
        # normalized: multiplying by the flag is what pins them to exactly 0.0.
        finite = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        values.append(torch.from_numpy(feature.normalize(finite) * flags))
        present.append(torch.from_numpy(flags))
    return torch.stack(tensors=values, dim=1), torch.stack(tensors=present, dim=1)
