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
from src.logs.keys import (
    ACTIVITY_KEY,
    EOT_TOKEN,
    EVENT_DELTA_KEY,
    PAD_TOKEN,
    REMAINING_TIME_KEY,
    RESOURCE_KEY,
    SOS_TOKEN,
    UNK_TOKEN,
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
    """One numeric channel of the log, and the log1p + min-max range it is normalized into.
    - clip_value: Values above this train-split percentile are clipped before the log.
    - log_min: The minimum of the log1p of the clipped train values.
    - log_max: The maximum of the log1p of the clipped train values.
    Log1p is used since the numeric columns may span several orders of magnitude.
    """

    column: str
    clip_value: float
    log_min: float
    log_max: float

    @classmethod
    def fit(cls, train: pd.DataFrame, *, column: str, percentile: float) -> NumericColumn:
        """Fit one channel's normalization range on the train split.

        Missing values take no part in the fit: a percentile over a column holding NaN is NaN, and
        a NaN clip value would send every value of the channel to NaN without raising anything.

        Args:
            train: The split every range is fit on, and the only one any range is ever fit on.
            column: Which column of it this channel reads.
            percentile: Values above this train-split percentile are clipped before the log.
        Returns:
            The channel, holding the range its values are normalized into `[0, 1]` with.
        Raises:
            ValueError: If the column holds no finite value, or a negative one - `normalize` clips
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
        return cls(
            column=column,
            clip_value=clip_value,
            log_min=float(log_values.min()),
            log_max=float(log_values.max()),
        )

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Map raw values into `[0, 1]`, clipping them to this range first.

        Clipping bounds the outliers and log1p then pulls in the long tail, so the bulk of the
        values do not all end up squeezed into the bottom of the range: the time columns and the
        amounts beside them span several orders of magnitude.

        Args:
            values: The raw values to normalize.
        Returns:
            The same values in `[0, 1]`, as float32.
        """
        log_values = np.log1p(np.clip(values, 0.0, self.clip_value))
        span = self.log_max - self.log_min
        # A split where every value is identical would divide by zero here.
        if span <= 0:
            return np.zeros_like(log_values, dtype=np.float32)
        return np.clip((log_values - self.log_min) / span, 0.0, 1.0).astype(np.float32)

    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        """Read normalized values back as the raw quantity they came from.

        Approximate at the top of the range: clipping is lossy, so anything that was above
        `clip_value` on the way in comes back as `clip_value`.
        """
        span = self.log_max - self.log_min
        return np.expm1(np.asarray(normalized, dtype=np.float64) * span + self.log_min)


class DatasetDescription(StrictModel):
    """The description of a dataset, fit on the train split and written beside the splits."""

    activity: CategoricalColumn
    resource: CategoricalColumn
    delta: NumericColumn
    remaining_time: NumericColumn

    # The columns the config names beside the four above, sorted by dtype into the two kinds.
    categorical_features: tuple[CategoricalColumn, ...]
    numeric_features: tuple[NumericColumn, ...]

    # Which dataset the splits belong to, and the maximum trace length they were preprocessed to.
    # Both come from the config rather than from the fit, so neither is written to `dataset.json`.
    dataset: str = Field(..., exclude=True)
    max_trace_length: int = Field(..., exclude=True)

    @property
    def num_feature_categories(self) -> int:
        """Rows in the table every categorical feature channel shares: one PAD, then a block
        per feature. 1 on a dataset with no categorical features, where no table is built."""
        return 1 + sum(feature.num_rows for feature in self.categorical_features)

    @classmethod
    def fit(cls, train: pd.DataFrame, *, data_config: DataConfig) -> DatasetDescription:
        """Fit the description on the train split.
        Args:
            train: The train split, as `pipelines/preprocess.py` holds it before writing.
            data_config: The `data` section, for the feature columns and the clip percentile.
        Returns:
            The description to write beside the splits.
        """
        percentile = data_config.time_clip_percentile
        categorical_features, numeric_features = _fit_event_features(
            train=train, columns=data_config.event_features, percentile=percentile
        )
        return cls(
            activity=CategoricalColumn.fit(
                train, column=ACTIVITY_KEY, special_tokens=ACTIVITY_TOKENS
            ),
            resource=CategoricalColumn.fit(
                train, column=RESOURCE_KEY, special_tokens=RESOURCE_TOKENS
            ),
            delta=NumericColumn.fit(train, column=EVENT_DELTA_KEY, percentile=percentile),
            remaining_time=NumericColumn.fit(
                train, column=REMAINING_TIME_KEY, percentile=percentile
            ),
            categorical_features=categorical_features,
            numeric_features=numeric_features,
            dataset=data_config.name,
            max_trace_length=data_config.max_seq_len,
        )

    @classmethod
    def load(cls, data_config: DataConfig) -> DatasetDescription:
        """Load the description previously generated for a dataset.

        The one place a config becomes a description: what comes back names the dataset and its
        sequence length as well as its encoding, so everything read from disk afterwards is asked
        of the description alone.

        Args:
            data_config: The `data` section, naming the dataset and the sequence length.
        Returns:
            The dataset description, with `dataset` and `max_trace_length` taken from the config.
        Raises:
            FileNotFoundError: If the dataset has not been preprocessed.
        """
        path = paths.description_path(data_config.name)
        if not path.exists():
            raise FileNotFoundError(
                f'no dataset description at {path}. Run `python -m pipelines.preprocess` first.'
            )
        return cls.model_validate(
            json.loads(path.read_text())
            | {'dataset': data_config.name, 'max_trace_length': data_config.max_seq_len}
        )

    def save(self) -> Path:
        """Write this description beside the splits, and return where it went."""
        path = paths.description_path(self.dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path

    def split_path(self, split: str) -> Path:
        """Where one preprocessed split of this dataset is kept.

        Args:
            split: Which of `train`, `val`, `test` to name.
        Returns:
            The path to that split's file.
        """
        return paths.split_path(dataset=self.dataset, split=split)


def _fit_event_features(
    *,
    train: pd.DataFrame,
    columns: list[str],
    percentile: float,
) -> tuple[tuple[CategoricalColumn, ...], tuple[NumericColumn, ...]]:
    """Sort the configured columns into categorical and numeric, and fit each on the train split.
    Args:
        train: The split every vocabulary and range is fit on.
        columns: The configured columns, in the order they will occupy the shared table and the
            embedding projection's input.
        percentile: Passed to `NumericColumn.fit` for each numeric column.
    Returns:
        The categorical features with their table offsets assigned, and the numeric features
        with their fitted ranges.
    """
    categorical: list[CategoricalColumn] = []
    numeric: list[NumericColumn] = []
    # Row 0 of the shared table is the PAD every feature uses, so the first block starts at 1.
    offset = 1

    for column in columns:
        if is_numeric_dtype(train[column]):
            numeric.append(NumericColumn.fit(train, column=column, percentile=percentile))
        else:
            feature = CategoricalColumn.fit(
                train, column=column, special_tokens=FEATURE_TOKENS, offset=offset
            )
            categorical.append(feature)
            # The next feature's block starts after this one's vocabulary and its UNK row.
            offset += feature.num_rows

    return tuple(categorical), tuple(numeric)
