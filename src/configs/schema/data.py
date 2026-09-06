from __future__ import annotations

from pydantic import Field, model_validator

from src.logs import CYCLE_TIME_KEY

from .base import NAME_PATTERN, StrictModel


class DataConfig(StrictModel):
    """How the raw log is read, preprocessed and split into train/val/test."""

    name: str = Field(..., pattern=NAME_PATTERN)

    case_key: str
    activity_key: str
    resource_key: str
    timestamp_key: str
    separator: str = Field(
        ...,
        min_length=1,
        description='Field separator of the raw log, as it was downloaded. Only that file: '
        'everything preprocessing writes uses `CSV_SEPARATOR`',
    )

    train_split: float = Field(..., gt=0.0, lt=1.0)
    val_split: float = Field(..., gt=0.0, lt=1.0)
    test_split: float = Field(..., gt=0.0, lt=1.0)

    max_seq_len_percentile: float = Field(
        ...,
        gt=0.0,
        le=100.0,
        description='Longer cases are dropped; the cutoff is what sequence tensors are padded to',
    )
    max_case_duration_percentile: float = Field(
        ...,
        gt=0.0,
        le=100.0,
        description='Longer-running cases are dropped before the split, so that a handful of '
        'broken timestamps does not set the statistics every duration channel is standardized '
        'against',
    )

    string_features: list[str] = Field(
        ..., description='Columns read as strings rather than by the dtype pandas infers'
    )
    event_features: list[str] = Field(
        ...,
        description='Per-event columns the encoders read beside the activity and the resource, '
        'either raw or derived from the timestamp (see `src/logs/keys.py`). A numeric one becomes '
        'a value and a present flag, anything else a vocabulary. Changing this list requires '
        're-preprocessing',
    )
    log_scaled_features: list[str] = Field(
        ..., description='Which `event_features` take a log1p before being standardized'
    )
    log_scaled_remaining_time: bool
    log_scaled_cycle_time: bool

    @model_validator(mode='after')
    def _splits_sum_to_one(self) -> DataConfig:
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f'train/val/test splits must sum to 1.0, got {total}')
        return self

    @model_validator(mode='after')
    def _log_scaled_features_are_event_features(self) -> DataConfig:
        # A column named here but not read as a feature would scale nothing, silently.
        unknown = sorted(set(self.log_scaled_features) - set(self.event_features))
        if unknown:
            raise ValueError(
                f'log_scaled_features names columns that are not event_features: '
                f'{", ".join(unknown)}'
            )
        return self

    @model_validator(mode='after')
    def _cycle_time_is_not_an_event_feature(self) -> DataConfig:
        # The cycle time would otherwise be fit twice: once here, once as `DatasetCodec.cycle_time`.
        if CYCLE_TIME_KEY in self.event_features:
            raise ValueError(
                f'event_features names "{CYCLE_TIME_KEY}", which is read through '
                f'DatasetCodec.cycle_time instead and must not also be listed here'
            )
        return self


class DeclareConfig(StrictModel):
    """Declarative-model discovery, run once per dataset on the train split."""

    consider_vacuity: bool = Field(
        ..., description='Whether a trace that never activates a constraint counts as satisfying it'
    )
    min_support: float = Field(
        ...,
        gt=0.0,
        le=1.0,
        description='Fraction of train traces a constraint must hold on to be kept',
    )
    itemsets_support: float = Field(
        ...,
        gt=0.0,
        le=1.0,
        description='Support floor for the frequent itemsets candidate constraints are built from',
    )
    max_cardinality: int = Field(
        ..., gt=0, description='Highest n tried for the templates that take one (`Existence3[A]`)'
    )
