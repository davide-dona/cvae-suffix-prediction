from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.configs.schema import DataConfig
from src.logs.io import read_log
from src.logs.keys import (
    CASE_KEY,
    CASE_OFFSET_KEY,
    EVENT_DELTA_KEY,
    ACTIVITY_KEY,
    MISSING_FEATURE,
    REMAINING_TIME_KEY,
    RESOURCE_KEY,
    TIMESTAMP_KEY,
)

# The columns an feature channel may not name: the two the model predicts, the two it already
# reads as channels of their own, and the three that identify or order an event.
_STRUCTURAL_COLUMNS = frozenset({
    CASE_KEY,
    ACTIVITY_KEY,
    RESOURCE_KEY,
    TIMESTAMP_KEY,
    CASE_OFFSET_KEY,
    EVENT_DELTA_KEY,
    REMAINING_TIME_KEY,
})


@dataclass(frozen=True)
class NormalizationRange:
    """log1p + min-max normalization range for one numeric column, fit on train.

    `clip_value` is the raw (pre-log1p), train-split percentile that values are clipped
    to before taking the log, so a handful of extreme outliers don't blow out the range
    the rest get normalized into. Both time columns span several orders of magnitude,
    which is what the log is for, and so do the amounts and counts the logs carry beside them.
    """
    clip_value: float
    log_min: float
    log_max: float


@dataclass(frozen=True)
class CategoricalFeature:
    """One log column read as a categorical event channel, and where it sits in the one
    embedding table every such channel shares.

    Row 0 of that table is a PAD shared by every feature, so `nn.Embedding(padding_idx=0)` pins
    the padding of all of them at once. Each feature then owns the `len(vocab) + 1` rows from
    `offset` on, the last of them being its own UNK.

    The order these are built in is the order they occupy the table, and so the order they
    occupy `EventEmbeddings.projection`'s input. Reordering a config's feature list without
    changing its length produces weights that load cleanly and mean something else.
    """
    column: str
    vocab: tuple[str, ...]
    offset: int

    @property
    def unk_index(self) -> int:
        """Every value val/test holds that the train split did not, collapsed into one row."""
        return self.offset + len(self.vocab)

    @property
    def num_rows(self) -> int:
        """Rows this feature owns in the shared table, its UNK included."""
        return len(self.vocab) + 1


@dataclass(frozen=True)
class NumericFeature:
    """One log column read as a numeric event channel: a normalized value and a present flag.

    The flag is how a missing value is told apart from a real 0.0, which matters on the columns
    a log sets only on part of its events - bpic-2017's offer amounts sit on ~4% of rows.

    Every numeric channel goes through the same log1p range as the two time columns. That is
    what the amounts and counts want, and on a bounded column like an age it is a harmless
    monotone squash with a learned projection immediately downstream.
    """
    column: str
    range: NormalizationRange


def as_categories(column: pd.Series) -> pd.Series:
    """One categorical channel's raw values as the strings its vocabulary is fit over.

    pandas leaves NA alone under `astype(str)`, so a column with gaps would carry a float NaN
    into a vocabulary of strings and not even sort. Missing becomes a token of its own instead,
    which is also what tells the model a value was absent rather than unseen.

    Both the fit in `_build_event_features` and the lookup in `Codec` go through here, which
    is what keeps the two agreeing.
    """
    return column.fillna(MISSING_FEATURE).astype(str)


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
    delta_stats: NormalizationRange
    remaining_time_stats: NormalizationRange

    # The columns this log offers beyond the three every event has, read by the encoders and
    # by nothing else. Empty tuples on a dataset whose config names none, which is what leaves
    # the model exactly the size it was without them.
    categorical_features: tuple[CategoricalFeature, ...]
    numeric_features: tuple[NumericFeature, ...]

    max_trace_length: int

    # The special tokens sit directly above the fitted vocabulary. Defining the layout here
    # rather than in `Codec` keeps one definition for the data layer and the model embeddings
    # to agree on.
    # Activities lay out (EOT, PAD, SOS, UNK): the decoder reads SOS as a real, embedded
    # start-of-sequence token.
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

    # Resources lay out (EOT, PAD, UNK): unlike activities, the decoder never reads a
    # resource, so there is no SOS token to reserve a row for.
    @property
    def num_resources(self) -> int:
        """Vocab size including the EOT, PAD and UNK tokens."""
        return len(self.resource_vocab) + 3

    @property
    def eot_resource_index(self) -> int:
        return len(self.resource_vocab)

    @property
    def pad_resource_index(self) -> int:
        return len(self.resource_vocab) + 1

    @property
    def unk_resource_index(self) -> int:
        """Every resource val/test holds that the train split did not, collapsed into one token."""
        return len(self.resource_vocab) + 2

    @property
    def num_feature_categories(self) -> int:
        """Rows in the table every categorical feature channel shares: one PAD, then a block
        per feature. 1 on a dataset with no categorical features, where no table is built."""
        return 1 + sum(feature.num_rows for feature in self.categorical_features)

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

        categorical_features, numeric_features = _build_event_features(
            train=train,
            val=val,
            test=test,
            columns=data_config.event_features,
            percentile=data_config.time_clip_percentile,
        )

        return cls(
            train=train,
            val=val,
            test=test,
            activity_vocab=activity_vocab,
            resource_vocab=resource_vocab,
            delta_stats=_fit_normalization(
                train, column=EVENT_DELTA_KEY, percentile=data_config.time_clip_percentile
            ),
            remaining_time_stats=_fit_normalization(
                train, column=REMAINING_TIME_KEY, percentile=data_config.time_clip_percentile
            ),
            categorical_features=categorical_features,
            numeric_features=numeric_features,
            max_trace_length=data_config.max_seq_len,
        )


def _build_event_features(
    *,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    percentile: float,
) -> tuple[tuple[CategoricalFeature, ...], tuple[NumericFeature, ...]]:
    """Sort the configured event-feature columns into the two channel kinds and fit each on train.

    The kind is the train split's dtype, applied unchanged to val/test. `read_log` lets pandas
    infer dtypes per split file, so a column read as numbers in one and as strings in another is
    an error raised here rather than a channel of UNKs nobody notices.

    Args:
        train: The split every vocabulary and range is fit on.
        val: Read only to check that it agrees with `train` about each column.
        test: The same.
        columns: The configured columns, in the order they will occupy the shared table and the
            embedding projection's input.
        percentile: Passed to `_fit_normalization` for each numeric column.
    Returns:
        The categorical features with their table offsets assigned, and the numeric features
        with their fitted ranges.
    Raises:
        ValueError: If a column is named twice, is missing from a split, names a structural
            column, or is numeric in one split and not in another.
    """
    duplicates = sorted({column for column in columns if columns.count(column) > 1})
    if duplicates:
        raise ValueError(
            f'event_features names {duplicates} more than once; each would become a channel of '
            'its own, holding the same values as the other'
        )

    categorical: list[CategoricalFeature] = []
    numeric: list[NumericFeature] = []
    # Row 0 of the shared table is the PAD every feature uses, so the first block starts at 1.
    offset = 1

    for column in columns:
        if column in _STRUCTURAL_COLUMNS:
            raise ValueError(
                f'event feature "{column}" is a structural column. The model already reads '
                'or predicts it; naming it here would feed it back as a channel of its own.'
            )
        for name, split in (('train', train), ('val', val), ('test', test)):
            if column not in split.columns:
                raise ValueError(f'event feature "{column}" is not in the {name} split')

        if is_numeric_dtype(train[column]):
            for name, split in (('val', val), ('test', test)):
                if not is_numeric_dtype(split[column]):
                    raise ValueError(
                        f'event feature "{column}" is numeric on train but not on {name}. '
                        'Give it a value on every row, or drop it from the config.'
                    )
            numeric.append(
                NumericFeature(
                    column=column,
                    range=_fit_normalization(train, column=column, percentile=percentile),
                )
            )
            continue

        for name, split in (('val', val), ('test', test)):
            if is_numeric_dtype(split[column]):
                raise ValueError(
                    f'event feature "{column}" is numeric on {name} but not on train. '
                    'Give it a value on every row, or drop it from the config.'
                )
        vocab = tuple(sorted(as_categories(train[column]).unique().tolist()))
        categorical.append(CategoricalFeature(column=column, vocab=vocab, offset=offset))
        offset += len(vocab) + 1

    return tuple(categorical), tuple(numeric)


def _fit_normalization(
    train: pd.DataFrame, *, column: str, percentile: float
) -> NormalizationRange:
    """Fit the log1p + min-max range of one numeric column on the train split.

    Missing values take no part in the fit: a percentile over a column holding NaN is NaN, and a
    NaN clip value would send every value of the channel to NaN without raising anything.

    Args:
        train: The train split, which is the only one any range is ever fit on.
        column: Which column to fit.
        percentile: Values above this train-split percentile are clipped before the log.
    Returns:
        The range `Codec` normalizes that column into `[0, 1]` with.
    Raises:
        ValueError: If the column holds no finite value, or a negative one - `_normalize` clips
            at 0.0 before the log, which would swallow it silently.
    """
    values = train[column].to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f'column "{column}" holds no finite value on the train split')
    if finite.min() < 0:
        raise ValueError(
            f'column "{column}" holds negative values (min {finite.min()}), which the log1p '
            'normalization would clip to 0 rather than represent'
        )

    clip_value = float(np.percentile(finite, percentile))
    log_values = np.log1p(np.clip(finite, 0.0, clip_value))
    return NormalizationRange(
        clip_value=clip_value,
        log_min=float(log_values.min()),
        log_max=float(log_values.max()),
    )
