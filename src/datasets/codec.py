from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pydantic import Field

from src import paths
from src.configs.schema import DataConfig, StrictModel
from src.logs import (
    ACTIVITY_KEY,
    CYCLE_TIME_KEY,
    EOT_TOKEN,
    PAD_TOKEN,
    REMAINING_TIME_KEY,
    RESOURCE_KEY,
    SOS_TOKEN,
    UNK_TOKEN,
    Split,
)

# The special tokens every channel carries, in the order they are indexed. They come before the
# vocabulary, so PAD is row 0 of the activity and resource channels, which start at offset 0.
# Row 0 of the shared feature table is a PAD too, so padding a run of events is a plain zero fill.
ACTIVITY_TOKENS = (PAD_TOKEN, EOT_TOKEN, SOS_TOKEN, UNK_TOKEN)
RESOURCE_TOKENS = (PAD_TOKEN, EOT_TOKEN, UNK_TOKEN)
FEATURE_TOKENS = (UNK_TOKEN,)


class CategoricalColumn(StrictModel):
    """One categorical channel of the log, and the vocabulary it is embedded through."""

    column: str
    vocab: tuple[str, ...]
    special_tokens: tuple[str, ...]
    offset: int

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        *,
        column: str,
        special_tokens: tuple[str, ...],
        offset: int = 0,
    ) -> CategoricalColumn:
        """Fit one channel's vocabulary on the train split.

        Args:
            train: The split every vocabulary is fit on.
            column: Which column of it this channel reads.
            special_tokens: The tokens following the vocabulary, in the order they are indexed.
            offset: Where this channel's block starts in the table it is embedded with.
        Returns:
            The channel, its every index derivable from the vocabulary it holds.
        """
        return cls(
            column=column,
            vocab=tuple(sorted(train[column].astype(str).unique().tolist())),
            special_tokens=special_tokens,
            offset=offset,
        )

    @property
    def num_rows(self) -> int:
        """Rows this channel owns in its table, its special tokens included."""
        return len(self.vocab) + len(self.special_tokens)

    @cached_property
    def to_index(self) -> dict[str, int]:
        """Each value of the train split to its row, the vocabulary following the special tokens.
        The special tokens are absent: no raw value maps to one, which is what makes an unseen
        value fall through to `unk_index`."""
        start = self.offset + len(self.special_tokens)
        return {value: start + i for i, value in enumerate(self.vocab)}

    @cached_property
    def from_index(self) -> dict[int, str]:
        """Each row back to the value it stands for, the special tokens included.

        UNK is one-way in spirit - many raw values map to it - but it still appears here, since
        reading a prediction back has to name the row somehow, and which value it was is exactly
        what the encoding threw away.
        """
        return {i: value for value, i in self.to_index.items()} | {
            self._index_of(token): token for token in self.special_tokens
        }

    @property
    def names(self) -> tuple[str, ...]:
        """Every value `decode` can name, in row order, the special tokens included.

        The one order a dataset's activities are coded in wherever a suffix is written as a string:
        seeding `ActivityCodes` from this is what makes the generations and the continuation index
        spell one suffix the same way, and it covers every row so no name is ever coded on the fly.
        """
        return tuple(self.from_index[row] for row in sorted(self.from_index))

    @property
    def eot_index(self) -> int:
        """The row marking the end of a trace."""
        return self._index_of(EOT_TOKEN)

    @property
    def pad_index(self) -> int:
        """The row filling a sequence out to `max_trace_length`, row 0 of any channel
        carrying it."""
        return self._index_of(PAD_TOKEN)

    @property
    def sos_index(self) -> int:
        """The row a generation starts from."""
        return self._index_of(SOS_TOKEN)

    @property
    def unk_index(self) -> int:
        """Every value val/test holds that the train split did not, collapsed into one row."""
        return self._index_of(UNK_TOKEN)

    def encode(self, log: pd.DataFrame) -> np.ndarray:
        """Map this channel's raw values to rows of the table it is embedded with, with an UNK
        for values the train split did not see.

        Args:
            log: The events as a preprocessed split holds them.
        Returns:
            `[len(log)]` of int64 rows.
        """
        return log[self.column].map(self.to_index).fillna(self.unk_index).to_numpy(dtype=np.int64)

    def decode(self, indices: np.ndarray, *, length: int) -> list[str]:
        """Read a run of this channel's indices back into the log's own values.

        Args:
            indices: The channel's indices for one sequence, int64, `[seq_len]`.
            length: How many of them to keep, the rest being padding.
        Returns:
            The values, in order.
        """
        return [self.from_index[int(index)] for index in indices[:length]]

    def _index_of(self, token: str) -> int:
        """The row of one special token.

        Raises:
            ValueError: If this channel does not carry that token - a resource is never generated,
                so it has no SOS, and asking for one is a bug rather than a missing default.
        """
        if token not in self.special_tokens:
            raise ValueError(f'column "{self.column}" carries no {token} token')
        return self.offset + self.special_tokens.index(token)


class NumericColumn(StrictModel):
    """One numeric channel of the log, and the train-split statistics it is standardized with.
    - log: Whether the values pass through a log1p before the standardization.
    - mean: The mean of the train values, after the log1p if there is one.
    - std: The standard deviation of the same, 1.0 for a channel that never varies.
    """

    column: str
    log: bool
    mean: float
    std: float

    @classmethod
    def fit(cls, train: pd.DataFrame, *, column: str, log: bool) -> NumericColumn:
        """Fit one channel's standardization on the train split.

        Missing values take no part in the fit, so a channel with gaps is standardized on the
        values it does have.

        Args:
            train: The split every statistic is fit on, and the only one any is ever fit on.
            column: Which column of it this channel reads.
            log: Whether to put the values through a log1p before taking mean and deviation.
        Returns:
            The channel, holding the statistics its values are standardized with.
        Raises:
            ValueError: If the column holds no finite value, or a negative one on a log-scaled
                channel, where log1p is undefined below -1 and compresses the rest towards it.
        """
        values = train[column].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(f'column "{column}" holds no finite value on the train split')
        if log and finite.min() < 0:
            raise ValueError(
                f'column "{column}" is log-scaled but holds negative values '
                f'(min {finite.min()}), which log1p cannot represent'
            )

        scaled = cls._scale(finite, log=log)
        # A channel whose train values are all identical has no deviation to divide by. 1.0
        # leaves every one of them standardizing to 0.0 and denormalizing back to the mean.
        std = float(scaled.std())
        return cls(
            column=column,
            log=log,
            mean=float(scaled.mean()),
            std=std if std > 0 else 1.0,
        )

    @staticmethod
    def _scale(values: np.ndarray, *, log: bool) -> np.ndarray:
        """Apply this channel's transform to the raw values."""
        return np.log1p(values) if log else values

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Standardize raw values against the train split's mean and deviation.

        Args:
            values: The raw values to normalize.
        Returns:
            The same values standardized, as float32. Nothing bounds them: a val/test value
            beyond anything the train split held keeps its distance rather than being pulled
            back to a range.
        """
        return ((self._scale(values, log=self.log) - self.mean) / self.std).astype(np.float32)

    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        """Read standardized values back as the raw quantity they came from, exactly: the
        transform loses nothing on the way in."""
        scaled = np.asarray(normalized, dtype=np.float64) * self.std + self.mean
        return np.expm1(scaled) if self.log else scaled

    def encode(self, log: pd.DataFrame) -> np.ndarray:
        """Standardize this channel's raw values, with a missing one pinned to exactly 0.0.

        Args:
            log: The events as a preprocessed split holds them.
        Returns:
            `[len(log)]` of float32.
        """
        raw = log[self.column].to_numpy(dtype=np.float64)
        # The fit skipped the gaps, so a gap is zeroed rather than standardized: multiplying by
        # the flag is what pins it to exactly 0.0, which is the train mean of the channel.
        finite = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        return self.normalize(finite) * self.present(log)

    def present(self, log: pd.DataFrame) -> np.ndarray:
        """Mark the rows this channel has a value for.

        Args:
            log: The events as a preprocessed split holds them.
        Returns:
            `[len(log)]` of float32, 0.0 where the log had no value. 0.0 is also a legitimate
            standardized value, which is what this flag is for.
        """
        return np.isfinite(log[self.column].to_numpy(dtype=np.float64)).astype(np.float32)


class DatasetCodec(StrictModel):
    """What a dataset's values are encoded through, fit on the train split and written beside
    the splits."""

    activity: CategoricalColumn
    resource: CategoricalColumn
    # cycle_time is read by both the encoders and the decoder; remaining_time is
    # decoder-only.
    cycle_time: NumericColumn
    remaining_time: NumericColumn

    # The columns `data.event_features` names, sorted by dtype into the two kinds.
    categorical_features: tuple[CategoricalColumn, ...]
    numeric_features: tuple[NumericColumn, ...]

    # The maximum trace length the splits were preprocessed to: `data.max_seq_len_percentile` of
    # case length, measured on the whole log. Written to `dataset.json`, since it is a property of
    # the fit rather than of the config, which only holds the percentile.
    max_trace_length: int

    # Which dataset the splits belong to. From the config rather than the fit, and excluded from
    # `dataset.json`, since it is the directory the file is written into.
    dataset: str = Field(..., exclude=True)

    @property
    def num_feature_categories(self) -> int:
        """Rows in the table every categorical feature channel shares: one PAD, then a block
        per feature. 1 on a dataset with no categorical features, where no table is built."""
        return 1 + sum(feature.num_rows for feature in self.categorical_features)

    @classmethod
    def fit(
        cls, train: pd.DataFrame, *, data_config: DataConfig, max_trace_length: int
    ) -> DatasetCodec:
        """Fit the codec on the train split.
        Args:
            train: The train split, as `pipelines/preprocess.py` holds it before writing.
            data_config: The `data` section, for the feature columns, which of them are
                log-scaled, and how the two time targets are scaled.
            max_trace_length: The sequence length splits were preprocessed to, from
                `pipelines.preprocess.case_length_cutoff`.
        Returns:
            The codec to write beside the splits.
        """
        categorical_features, numeric_features = _fit_event_features(
            train=train,
            columns=data_config.event_features,
            log_scaled=set(data_config.log_scaled_features),
        )
        return cls(
            activity=CategoricalColumn.fit(
                train, column=ACTIVITY_KEY, special_tokens=ACTIVITY_TOKENS
            ),
            resource=CategoricalColumn.fit(
                train, column=RESOURCE_KEY, special_tokens=RESOURCE_TOKENS
            ),
            cycle_time=NumericColumn.fit(
                train, column=CYCLE_TIME_KEY, log=data_config.log_scaled_cycle_time
            ),
            remaining_time=NumericColumn.fit(
                train, column=REMAINING_TIME_KEY, log=data_config.log_scaled_remaining_time
            ),
            categorical_features=categorical_features,
            numeric_features=numeric_features,
            dataset=data_config.name,
            max_trace_length=max_trace_length,
        )

    @classmethod
    def load(cls, data_config: DataConfig) -> DatasetCodec:
        """Load the codec previously fit for a dataset.

        The one place a config becomes a codec: what comes back names the dataset, so everything
        read from disk afterwards is asked of the codec alone.

        Args:
            data_config: The `data` section, naming the dataset.
        Returns:
            The dataset codec, with `dataset` taken from the config and everything else,
            `max_trace_length` included, read back from what preprocessing fit.
        Raises:
            FileNotFoundError: If the dataset has not been preprocessed.
        """
        name = data_config.name
        path = paths.CODEC.require(name)
        return cls.model_validate(json.loads(path.read_text()) | {'dataset': name})

    def save(self) -> Path:
        """Write this codec to its own directory, and return where it went."""
        path = paths.CODEC.prepare(self.dataset)
        path.write_text(self.model_dump_json(indent=2))
        return path

    def split_path(self, split: Split) -> Path:
        """Where one preprocessed split of this dataset is kept.

        Args:
            split: Which of the three to name.
        Returns:
            The path to that split's file.
        """
        return paths.PROCESSED_SPLIT.path(dataset=self.dataset, split=split)


def _fit_event_features(
    *,
    train: pd.DataFrame,
    columns: list[str],
    log_scaled: set[str],
) -> tuple[tuple[CategoricalColumn, ...], tuple[NumericColumn, ...]]:
    """Sort the configured columns into categorical and numeric, and fit each on the train split.
    Args:
        train: The split every vocabulary and statistic is fit on.
        columns: The configured columns, in the order they will occupy the shared table and the
            embedding projection's input.
        log_scaled: Which of them take a log1p before the standardization. A name in here that
            turns out categorical simply never comes up, the column having nothing to scale.
    Returns:
        The categorical features with their table offsets assigned, and the numeric features
        with their fitted statistics.
    """
    categorical: list[CategoricalColumn] = []
    numeric: list[NumericColumn] = []
    # Row 0 of the shared table is the PAD every feature uses, so the first block starts at 1.
    offset = 1

    for column in columns:
        if is_numeric_dtype(train[column]):
            numeric.append(NumericColumn.fit(train, column=column, log=column in log_scaled))
        else:
            feature = CategoricalColumn.fit(
                train, column=column, special_tokens=FEATURE_TOKENS, offset=offset
            )
            categorical.append(feature)
            # The next feature's block starts after this one's vocabulary and its UNK row.
            offset += feature.num_rows

    return tuple(categorical), tuple(numeric)
