from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pydantic import Field

from src.configs.schema import DataConfig, StrictModel
from src.logs.io import read_log
from src.logs.keys import (
    ACTIVITY_KEY,
    CASE_KEY,
    EVENT_DELTA_KEY,
    MISSING_FEATURE,
    REMAINING_TIME_KEY,
    RESOURCE_KEY,
)


class NormalizationRange(StrictModel):
    """log1p + min-max normalization range for one numeric column, fit on train.
    - clip_value: Values above this train-split percentile are clipped before the log.
    - log_min: The minimum of the log1p of the clipped train values.
    - log_max: The maximum of the log1p of the clipped train values.
    Log1p is used since the numeric columns may span several orders of magnitude.
    """
    clip_value: float
    log_min: float
    log_max: float


class CategoricalFeature(StrictModel):
    """One log column categorical feature: a vocabulary and its offset in the shared table."""
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


class NumericFeature(StrictModel):
    """One log column numeric feature: the column name and its normalization range."""
    column: str
    range: NormalizationRange


def as_categories(column: pd.Series) -> pd.Series:
    """One categorical channel's raw values as the strings its vocabulary is fit over.

    pandas leaves NA alone under `astype(str)`, so a column with gaps would carry a float NaN
    into a vocabulary of strings and not even sort. Missing becomes a token of its own instead,
    which is also what tells the model a value was absent rather than unseen.

    Both the fit in `_fit_event_features` and the lookup in `Codec` go through here, which
    is what keeps the two agreeing.
    """
    return column.fillna(MISSING_FEATURE).astype(str)


class DatasetInfo(StrictModel):
    """Everything `Codec`, `SuffixDataset` and the model need to know about a dataset.

    Fit once by `pipelines/preprocess.py`, from the train split only, and written next to the
    splits; `load` reads it back. Val and test are never looked at, so nothing about them can
    leak into a vocabulary or a range.

    This is also the single source of the data-derived dimensions the model is built
    against (vocabulary sizes, special-token indices, sequence length), so `TransformerCVAE`
    needs nothing beyond `ModelConfig` and one of these.
    """

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
    # the model exactly the size it was without them. The order they are fit in is the order
    # they occupy the shared table and `EventEmbeddings.projection`'s input.
    categorical_features: tuple[CategoricalFeature, ...]
    numeric_features: tuple[NumericFeature, ...]

    # Not written to the file: unlike everything above it is a config knob rather than
    # something fit from the log, so `load` takes it from `data.max_seq_len` and changing it
    # does not mean preprocessing again.
    max_trace_length: int = Field(..., exclude=True)

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
    def fit(cls, train: pd.DataFrame, *, data_config: DataConfig) -> "DatasetInfo":
        """Fit every vocabulary and normalization range on the train split.

        Args:
            train: The train split, as `pipelines/preprocess.py` holds it before writing.
            data_config: The `data` section, for the feature columns and the clip percentile.
        Returns:
            The description to write beside the splits.
        """
        categorical_features, numeric_features = _fit_event_features(
            train=train,
            columns=data_config.event_features,
            percentile=data_config.time_clip_percentile,
        )
        return cls(
            activity_vocab=sorted(train[ACTIVITY_KEY].astype(str).unique().tolist()),
            resource_vocab=sorted(train[RESOURCE_KEY].astype(str).unique().tolist()),
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

    @classmethod
    def load(cls, data_config: DataConfig) -> "DatasetInfo":
        """Read back what `pipelines/preprocess.py` fit for this dataset.

        Args:
            data_config: The `data` section, naming the dataset directory and the sequence length.
        Returns:
            The dataset description, with `max_trace_length` taken from the config.
        Raises:
            FileNotFoundError: If the dataset has not been preprocessed.
        """
        path = metadata_path(data_config)
        if not path.exists():
            raise FileNotFoundError(
                f'no dataset description at {path}. Run `python -m pipelines.preprocess` first.'
            )
        return cls.model_validate(
            json.loads(path.read_text()) | {'max_trace_length': data_config.max_seq_len}
        )

    def save(self, data_config: DataConfig) -> Path:
        """Write this description beside the splits, and return where it went."""
        path = metadata_path(data_config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path


def metadata_path(data_config: DataConfig) -> Path:
    """Where a dataset's fitted description is kept, next to the splits it was fit on."""
    return data_config.dir / 'processed' / 'dataset.json'


def read_split(data_config: DataConfig, *, split: str, info: DatasetInfo) -> pd.DataFrame:
    """Read one preprocessed split, with its categorical columns forced to strings.

    Each split is a file of its own, so pandas would otherwise infer dtypes for each
    separately and a column of numeric-looking codes could come back as float in one and as
    str in another. Reading the columns a vocabulary was fit over as text is what keeps
    every split agreeing with that vocabulary.

    Args:
        data_config: The `data` section, naming the dataset directory.
        split: Which of `train`, `val`, `test` to read.
        info: The dataset description, for the categorical feature columns.
    Returns:
        The split, one row per event.
    """
    text_columns = {CASE_KEY: str, ACTIVITY_KEY: str, RESOURCE_KEY: str} | {
        feature.column: str for feature in info.categorical_features
    }
    return read_log(data_config.dir / 'processed' / f'{split}.csv', dtype=text_columns)


def _fit_event_features(
    *,
    train: pd.DataFrame,
    columns: list[str],
    percentile: float,
) -> tuple[tuple[CategoricalFeature, ...], tuple[NumericFeature, ...]]:
    """Sort the configured event-feature columns into the two channel kinds and fit each on train.

    The kind is the column's dtype: numeric columns become a value and a present flag, anything
    else a vocabulary.

    Args:
        train: The split every vocabulary and range is fit on.
        columns: The configured columns, in the order they will occupy the shared table and the
            embedding projection's input.
        percentile: Passed to `_fit_normalization` for each numeric column.
    Returns:
        The categorical features with their table offsets assigned, and the numeric features
        with their fitted ranges.
    """
    categorical: list[CategoricalFeature] = []
    numeric: list[NumericFeature] = []
    # Row 0 of the shared table is the PAD every feature uses, so the first block starts at 1.
    offset = 1

    for column in columns:
        if is_numeric_dtype(train[column]):
            numeric.append(
                NumericFeature(
                    column=column,
                    range=_fit_normalization(train, column=column, percentile=percentile),
                )
            )
            continue

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
