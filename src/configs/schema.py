from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model for all config sections: immutable and typo-proof."""

    model_config = ConfigDict(frozen=True, extra='forbid')


class DataConfig(StrictModel):
    """How the raw log is read, preprocessed and split into train/val/test."""

    name: str = Field(
        ...,
        description='Dataset identifier; every path under `data/` and `outputs/` derives from '
        "it. Filled in from the config's filename by `load_config`, not written in YAML",
    )

    case_key: str = Field(..., description='Raw column identifying the case each event belongs to')
    activity_key: str = Field(..., description='Raw column identifying the activity label')
    resource_key: str = Field(..., description='Raw column identifying the resource')
    timestamp_key: str = Field(..., description='Raw column holding the event timestamp')

    train_split: float = Field(..., gt=0.0, lt=1.0)
    val_split: float = Field(..., gt=0.0, lt=1.0)
    test_split: float = Field(..., gt=0.0, lt=1.0)

    split_strategy: Literal['temporal', 'uedlstm_replicated'] = Field(
        ...,
        description='How cases are assigned to splits. `temporal` cuts by case start time; '
        "`uedlstm_replicated` reproduces U-ED-LSTM's seeded shuffle, for comparable evaluation",
    )

    max_seq_len_percentile: float = Field(
        ...,
        gt=0.0,
        le=100.0,
        description='Cases longer than this percentile of case length are dropped at '
        'preprocessing time. The cutoff is what sequence tensors are padded to',
    )

    event_features: list[str] = Field(
        ...,
        description='Extra per-event columns the encoders read beside activity, resource and '
        'timestamps. A numeric one becomes a value and a present flag, anything else a '
        'vocabulary. Changing this list requires re-preprocessing the dataset',
    )

    log_scaled_features: list[str] = Field(
        ...,
        description='Which `event_features` take a log1p before being standardized, for columns '
        'spanning several orders of magnitude. Empty matches the baselines, which standardize '
        'every numeric column raw',
    )

    log_scaled_durations: bool = Field(
        ...,
        description='Whether the two timestamp proxies (`Events.ts_prev`, `Events.ts_start`) and '
        'remaining time take a log1p before being standardized. Off matches the baselines, which '
        'standardize durations raw',
    )

    batch_size: int = Field(..., gt=0)
    num_workers: int = Field(..., ge=0)

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


class DeclareConfig(StrictModel):
    """Declarative-model discovery, run once per dataset at preprocessing time on the train
    split only.
    """

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
        description='Support floor for the frequent activity itemsets candidate constraints '
        'are built from',
    )
    max_cardinality: int = Field(
        ..., gt=0, description='Highest n tried for the templates that take one (`Existence3[A]`)'
    )


class EmbeddingConfig(StrictModel):
    """Event embeddings, shared by both trace encoders and the decoder."""

    activity_dim: int = Field(..., gt=0)
    resource_dim: int = Field(..., gt=0)
    feature_dim: int = Field(
        ...,
        gt=0,
        description="Width of each categorical event feature's embedding table, shared across "
        'all of them: the more features, the more a bigger value widens the projection input',
    )


class TraceEncoderConfig(StrictModel):
    """Transformer encoder over a sequence of events, full self-attention.

    Used for both the prefix, whose summary conditions the prior and the decoder, and, on the
    training path only, the ground-truth suffix, which feeds the posterior. Runs at
    `ModelConfig.d_model`.
    """

    num_layers: int = Field(..., gt=0)
    num_heads: int = Field(
        ..., gt=0, description='Attention heads per layer; must divide `d_model`'
    )
    feedforward_dim: int = Field(
        ..., gt=0, description='Width of the feed-forward block inside a layer'
    )
    dropout: float = Field(..., ge=0.0, lt=1.0)


class PriorConfig(StrictModel):
    """MLP mapping the prefix summary to p(z | prefix), in place of the fixed N(0, I) prior of
    an unconditional VAE.
    """

    hidden_dims: list[int] = Field(..., description='Hidden layer widths; empty for a linear prior')
    dropout: float = Field(..., ge=0.0, lt=1.0)


class LatentConfig(StrictModel):
    latent_dim: int = Field(..., gt=0)


class DecoderConfig(StrictModel):
    """Transformer decoder writing the suffix: causal self-attention plus cross-attention over
    the encoded prefix. Runs at `ModelConfig.d_model`.
    """

    num_layers: int = Field(..., gt=0)
    num_heads: int = Field(
        ..., gt=0, description='Attention heads per layer; must divide `d_model`'
    )
    feedforward_dim: int = Field(
        ..., gt=0, description='Width of the feed-forward block inside a layer'
    )
    dropout: float = Field(..., ge=0.0, lt=1.0)
    activity_dropout: float = Field(
        ...,
        ge=0.0,
        lt=1.0,
        description='Fraction of teacher-forced activities blanked to PAD during training, '
        'forcing information into z instead. 0.0 disables it',
    )
    head_hidden_dim: int = Field(
        ..., gt=0, description='Width of the layer shared by the two output heads'
    )


class ModelConfig(StrictModel):
    """Every hyperparameter of `TransformerCVAE`.

    Data-derived dimensions (vocabulary sizes, special-token indices, sequence length) are
    absent: they come from `DatasetCodec` at build time.
    """

    d_model: int = Field(
        ..., gt=0, description='Shared width for the embeddings, both encoders and the decoder'
    )

    embeddings: EmbeddingConfig
    prefix_encoder: TraceEncoderConfig
    suffix_encoder: TraceEncoderConfig
    prior: PriorConfig
    latent: LatentConfig
    decoder: DecoderConfig

    @model_validator(mode='after')
    def _heads_divide_width(self) -> ModelConfig:
        # nn.MultiheadAttention asserts this when the layer is built, halfway through a run's
        # setup. Checking it here turns a config mistake back into a config error.
        for name in ('prefix_encoder', 'suffix_encoder', 'decoder'):
            num_heads = getattr(self, name).num_heads
            if self.d_model % num_heads != 0:
                raise ValueError(
                    f'model.{name}.num_heads ({num_heads}) must divide '
                    f'model.d_model ({self.d_model})'
                )
        return self


class LossConfig(StrictModel):
    """KL weighting: the annealing schedule and the free-bits floor.

    Both are measured in optimizer steps, not epochs, so they mean the same thing on every
    dataset. The cycle is a length rather than a count fitted into `training.max_steps`, so a
    run that stops early has still seen whole cycles.
    """

    kl_annealing_period_steps: int = Field(..., gt=0, description='Optimizer steps in one cycle')
    kl_annealing_ratio: float = Field(
        ..., gt=0.0, lt=1.0, description='Fraction of each cycle spent ramping up'
    )
    kl_annealing_start_weight: float = Field(
        ..., ge=0.0, description='Weight each cycle ramps up from'
    )
    kl_annealing_full_weight: float = Field(
        ..., ge=0.0, description='Weight each cycle ramps up to, and holds at'
    )

    free_bits: float = Field(
        ...,
        ge=0.0,
        description='Nats per latent dimension the KL is not penalized below. 0.0 leaves it '
        'unfloored',
    )


class OptimizerConfig(StrictModel):
    lr: float = Field(..., gt=0.0)
    weight_decay: float = Field(..., ge=0.0)


class TrainingConfig(StrictModel):
    """How long a run goes on for and how often it validates, both counted in optimizer steps
    rather than epochs, since an epoch is a different amount of learning on every log.
    """

    max_steps: int = Field(
        ...,
        gt=0,
        description='Ceiling on optimizer steps; early stopping normally ends a run first',
    )
    grad_clip_norm: float | None = Field(
        None, gt=0.0, description='Max gradient norm; null or absent leaves gradients unclipped'
    )
    device: Literal['cpu', 'cuda', 'mps']
    val_every_n_steps: int = Field(
        ...,
        gt=0,
        description='Steps between validations; also the unit `early_stopping.patience` counts in',
    )
    validation_pairs: int = Field(
        ...,
        gt=0,
        description='Teacher-forced validation pairs read per validation, capped by split size',
    )
    generation_pairs: int = Field(
        ...,
        gt=0,
        description='Free-running generation prefixes per validation; the sample the selection '
        'score is computed on',
    )


class InferenceConfig(StrictModel):
    """Generating suffixes for a whole split, which is what evaluation reads. Runs on
    `training.device`.
    """

    num_samples: int = Field(
        ...,
        ge=10,
        description='Suffixes generated per prefix from p(z | prefix). Minimum 10: '
        '`hit_rate_at_10` reads the tenth draw',
    )
    generation_rows: int = Field(
        ...,
        gt=0,
        description='Rows put through the decoder per call; bounds memory, since each row '
        'keeps a key/value cache per position and layer',
    )


class EarlyStoppingConfig(StrictModel):
    """Stops training once the free-running generation score plateaus, the same score used for
    best-model selection and reported by `pipelines/evaluate.py`.
    """

    patience: int = Field(
        ..., gt=0, description='Non-improving validations tolerated before stopping'
    )
    min_delta_perc: float = Field(
        ..., ge=0.0, description='Minimum relative improvement to reset the patience counter'
    )


class ExperimentConfig(StrictModel):
    """Top-level config, the single object loaded from YAML.

    Pass sub-sections (e.g. `cfg.model.encoder`, `cfg.optimizer`) into functions rather than
    this whole object, so each function only depends on the parameters it actually uses.
    """

    seed: int
    experiment_name: str

    data: DataConfig
    declare: DeclareConfig
    model: ModelConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    early_stopping: EarlyStoppingConfig
    inference: InferenceConfig
