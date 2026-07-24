from __future__ import annotations
import numpy as np

from src.configs.dataset_info import CategoricalConditionStats, DatasetInfo
from src.logs.keys import EOT_ACTIVITY, EOT_RESOURCE, PADDING_ACTIVITY, PADDING_RESOURCE


class Encoding:
    def __init__(self, dataset_info: DatasetInfo):
        self.activity_to_index = {activity: i for i, activity in enumerate(dataset_info.activity_vocab)}
        self.eot_activity_index = len(dataset_info.activity_vocab)
        self.pad_activity_index = len(dataset_info.activity_vocab) + 1
        self.index_to_activity = {i: a for a, i in self.activity_to_index.items()} | {
            self.eot_activity_index: EOT_ACTIVITY,
            self.pad_activity_index: PADDING_ACTIVITY,
        }

        self.resource_to_index = {resource: i for i, resource in enumerate(dataset_info.resource_vocab)}
        self.eot_resource_index = len(dataset_info.resource_vocab)
        self.pad_resource_index = len(dataset_info.resource_vocab) + 1
        self.index_to_resource = {i: r for r, i in self.resource_to_index.items()} | {
            self.eot_resource_index: EOT_RESOURCE,
            self.pad_resource_index: PADDING_RESOURCE,
        }

        self.s2i: dict[str, dict[str, int]] = {}

        self._time_stats = dataset_info.time_stats
        self._condition_stats = dataset_info.condition_stats

    def encode_activity(self, activity: str) -> int:
        """Look up an activity's index. Raises `KeyError` if it wasn't in the train vocab."""
        try:
            return self.activity_to_index[activity]
        except KeyError:
            raise KeyError(
                f"activity {activity!r} was not seen in the train split; Encoding is fit "
                "train-only and does not support unseen activities at val/test time"
            ) from None

    def encode_resource(self, resource: str) -> int:
        """Look up a resource's index. Raises `KeyError` if it wasn't in the train vocab."""
        try:
            return self.resource_to_index[resource]
        except KeyError:
            raise KeyError(
                f"resource {resource!r} was not seen in the train split; Encoding is fit "
                "train-only and does not support unseen resources at val/test time"
            ) from None

    def normalize_time(self, delta_minutes: np.ndarray) -> np.ndarray:
        """log1p + min-max into `[0, 1]`, clipping to the range fit on the train split."""
        stats = self._time_stats
        clipped = np.clip(delta_minutes, 0.0, stats.clip_value)
        log_delta = np.log1p(clipped)
        span = stats.log_max - stats.log_min
        if span <= 0:
            return np.zeros_like(log_delta, dtype=np.float32)
        return np.clip((log_delta - stats.log_min) / span, 0.0, 1.0).astype(np.float32)

    def denormalize_time(self, normalized: np.ndarray) -> np.ndarray:
        """Invert `normalize_time`. Approximate for deltas that were clipped going in."""
        stats = self._time_stats
        span = stats.log_max - stats.log_min
        log_delta = np.asarray(normalized, dtype=np.float64) * span + stats.log_min
        return np.expm1(log_delta)

    def condition_dim(self) -> int:
        """Width of the vector `encode_condition` returns."""
        total = 0
        for stats in self._condition_stats.values():
            total += len(stats.categories) if isinstance(stats, CategoricalConditionStats) else 1
        return total

    def encode_condition(self, values: dict[str, object]) -> np.ndarray:
        """One-hot encode categorical condition features and min-max scale numeric ones,
        concatenating them (in `condition_features` order) into one fixed-width vector.
        """
        parts = []
        for feature, stats in self._condition_stats.items():
            raw = values[feature]
            if isinstance(stats, CategoricalConditionStats):
                one_hot = np.zeros(len(stats.categories), dtype=np.float32)
                try:
                    one_hot[stats.categories.index(str(raw))] = 1.0
                except ValueError:
                    raise KeyError(
                        f"condition feature {feature!r} value {raw!r} was not seen in the "
                        "train split; Encoding is fit train-only"
                    ) from None
                parts.append(one_hot)
            else:
                span = stats.max - stats.min
                normalized = 0.0 if span <= 0 else np.clip((float(raw) - stats.min) / span, 0.0, 1.0)
                parts.append(np.array([normalized], dtype=np.float32))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
