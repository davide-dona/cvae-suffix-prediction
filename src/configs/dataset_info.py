from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.configs.schema import DataConfig
from src.logs.io import read_log
from src.logs.keys import EVENT_DELTA_KEY, ACTIVITY_KEY, RESOURCE_KEY


@dataclass(frozen=True)
class TimeStats:
    """log1p + min-max normalization stats for the event time-delta, fit on train.

    `clip_value` is the raw (pre-log1p), train-split percentile that deltas are clipped
    to before taking the log, so a handful of extreme outliers don't blow out the range
    the rest of the deltas get normalized into.
    """
    clip_value: float
    log_min: float
    log_max: float


@dataclass(frozen=True)
class CategoricalConditionStats:
    """Vocabulary of a categorical condition feature, fit on train."""
    categories: list[str]


@dataclass(frozen=True)
class NumericConditionStats:
    """Min-max normalization range of a numeric condition feature, fit on train."""
    min: float
    max: float


@dataclass(frozen=True)
class DatasetInfo:
    """Everything `Encoding` and `GenericDataset` need to know about a dataset.

    Build with `DatasetInfo.build(data_config)`. All vocabularies and normalization
    ranges are fit on `train` only and then reused, unchanged, for `val`/`test`.
    """
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    activity_vocab: list[str]
    resource_vocab: list[str]
    time_stats: TimeStats

    condition_features: list[str]
    condition_stats: dict[str, CategoricalConditionStats | NumericConditionStats]

    max_trace_length: int

    @property
    def num_activities(self) -> int:
        """Vocab size including the EOT and PAD tokens."""
        return len(self.activity_vocab) + 2

    @property
    def num_resources(self) -> int:
        """Vocab size including the EOT and PAD tokens."""
        return len(self.resource_vocab) + 2

    @classmethod
    def build(cls, data_config: DataConfig) -> "DatasetInfo":
        train = read_log(data_config.dir / "train.csv")
        val = read_log(data_config.dir / "val.csv")
        test = read_log(data_config.dir / "test.csv")

        activity_vocab = sorted(train[ACTIVITY_KEY].unique().tolist())
        resource_vocab = sorted(train[RESOURCE_KEY].astype(str).unique().tolist())

        time_stats = _fit_time_stats(train, percentile=data_config.time_clip_percentile)
        condition_stats = _fit_condition_stats(train, data_config.condition_features)

        return cls(
            train=train,
            val=val,
            test=test,
            activity_vocab=activity_vocab,
            resource_vocab=resource_vocab,
            time_stats=time_stats,
            condition_features=data_config.condition_features,
            condition_stats=condition_stats,
            max_trace_length=data_config.max_seq_len,
        )


def _fit_time_stats(train: pd.DataFrame, *, percentile: float) -> TimeStats:
    deltas = train[EVENT_DELTA_KEY].to_numpy(dtype=np.float64)
    clip_value = float(np.percentile(deltas, percentile))
    clipped = np.clip(deltas, 0.0, clip_value)
    log_deltas = np.log1p(clipped)
    return TimeStats(clip_value=clip_value, log_min=float(log_deltas.min()), log_max=float(log_deltas.max()))


def _fit_condition_stats(
    train: pd.DataFrame, condition_features: list[str]
) -> dict[str, CategoricalConditionStats | NumericConditionStats]:
    """Fit one stats object per condition feature.

    Condition features are case-level (constant within a case, see `GenericDataset`), so
    computing the vocabulary / range over every row rather than one row per case gives the
    same result and avoids an extra group-by here.
    """
    stats: dict[str, CategoricalConditionStats | NumericConditionStats] = {}
    for feature in condition_features:
        column = train[feature]
        if pd.api.types.is_numeric_dtype(column):
            stats[feature] = NumericConditionStats(min=float(column.min()), max=float(column.max()))
        else:
            stats[feature] = CategoricalConditionStats(categories=sorted(column.astype(str).unique().tolist()))
    return stats
