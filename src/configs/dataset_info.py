from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.configs.schema import DataConfig
from src.logs.io import read_log
from src.logs.keys import EVENT_DELTA_KEY, ACTIVITY_KEY, REMAINING_TIME_KEY, RESOURCE_KEY


@dataclass(frozen=True)
class TimeStats:
    """log1p + min-max normalization stats for one column of minutes, fit on train.

    `clip_value` is the raw (pre-log1p), train-split percentile that values are clipped
    to before taking the log, so a handful of extreme outliers don't blow out the range
    the rest get normalized into. Both time columns span several orders of magnitude,
    which is what the log is for.
    """
    clip_value: float
    log_min: float
    log_max: float


@dataclass(frozen=True)
class DatasetInfo:
    """Everything `Codec`, `SuffixDataset` and the model need to know about a dataset.

    Build with `DatasetInfo.build(data_config)`. All vocabularies and normalization
    ranges are fit on `train` only and then reused, unchanged, for `val`/`test`.

    This is also the single source of the data-derived dimensions the model is built
    against (vocabulary sizes, special-token indices, sequence length), so `TransformerCVAE`
    needs nothing beyond `ModelConfig` and one of these.
    """
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    activity_vocab: list[str]
    resource_vocab: list[str]
    # Two ranges, because the two columns are not the same quantity: a gap between
    # consecutive events, which the encoders read, and a whole case's time left to run,
    # which the model predicts. A sum is far larger than its terms, so one range fit on
    # the deltas would push every remaining time into the top of [0, 1].
    delta_stats: TimeStats
    remaining_time_stats: TimeStats

    max_trace_length: int

    # The special tokens sit directly above the fitted vocabulary, in this order (EOT, PAD,
    # SOS, UNK), for both activities and resources. Defining the layout here rather than in
    # `Codec` keeps one definition for the data layer and the model embeddings to agree on.
    @property
    def num_activities(self) -> int:
        """Vocab size including the EOT, PAD, SOS and UNK tokens."""
        return len(self.activity_vocab) + 4

    @property
    def eot_activity_index(self) -> int:
        return len(self.activity_vocab)

    @property
    def pad_activity_index(self) -> int:
        return len(self.activity_vocab) + 1

    @property
    def sos_activity_index(self) -> int:
        return len(self.activity_vocab) + 2

    @property
    def unk_activity_index(self) -> int:
        """Every activity val/test holds that the train split did not, collapsed into one token."""
        return len(self.activity_vocab) + 3

    @property
    def num_resources(self) -> int:
        """Vocab size including the EOT, PAD, SOS and UNK tokens."""
        return len(self.resource_vocab) + 4

    @property
    def eot_resource_index(self) -> int:
        return len(self.resource_vocab)

    @property
    def pad_resource_index(self) -> int:
        return len(self.resource_vocab) + 1

    @property
    def sos_resource_index(self) -> int:
        return len(self.resource_vocab) + 2

    @property
    def unk_resource_index(self) -> int:
        """Every resource val/test holds that the train split did not, collapsed into one token."""
        return len(self.resource_vocab) + 3

    @classmethod
    def build(cls, data_config: DataConfig) -> "DatasetInfo":
        # `pipelines/preprocess.py` writes the splits into this subdirectory.
        processed_dir = data_config.dir / "processed"
        train = read_log(processed_dir / "train.csv")
        val = read_log(processed_dir / "val.csv")
        test = read_log(processed_dir / "test.csv")

        # Both columns are stringified before the vocabulary is fit, and again on the lookup
        # side in `_group_traces`. `read_log` lets pandas infer dtypes per split file, so a
        # column of numeric-looking codes can come back as float in one split and str in
        # another; coercing both ends is what keeps the two agreeing.
        activity_vocab = sorted(train[ACTIVITY_KEY].astype(str).unique().tolist())
        resource_vocab = sorted(train[RESOURCE_KEY].astype(str).unique().tolist())

        return cls(
            train=train,
            val=val,
            test=test,
            activity_vocab=activity_vocab,
            resource_vocab=resource_vocab,
            delta_stats=_fit_time_stats(
                train, column=EVENT_DELTA_KEY, percentile=data_config.time_clip_percentile
            ),
            remaining_time_stats=_fit_time_stats(
                train, column=REMAINING_TIME_KEY, percentile=data_config.time_clip_percentile
            ),
            max_trace_length=data_config.max_seq_len,
        )


def _fit_time_stats(train: pd.DataFrame, *, column: str, percentile: float) -> TimeStats:
    """Fit the log1p + min-max range of one column of minutes on the train split.

    Args:
        train: The train split, which is the only one any range is ever fit on.
        column: Which column of minutes to fit, `EVENT_DELTA_KEY` or `REMAINING_TIME_KEY`.
        percentile: Values above this train-split percentile are clipped before the log.
    Returns:
        The range `Codec` normalizes that column into `[0, 1]` with.
    """
    minutes = train[column].to_numpy(dtype=np.float64)
    clip_value = float(np.percentile(minutes, percentile))
    log_minutes = np.log1p(np.clip(minutes, 0.0, clip_value))
    return TimeStats(
        clip_value=clip_value,
        log_min=float(log_minutes.min()),
        log_max=float(log_minutes.max()),
    )
