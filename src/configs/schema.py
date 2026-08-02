from __future__ import annotations
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Strict base model for all config sections: immutable and typo-proof."""
    model_config = ConfigDict(frozen=True, extra="forbid")


class DataConfig(StrictModel):
    """Per-dataset config: one YAML per dataset, so paths and raw column
    names live here rather than as global constants."""

    dir: Path = Field(..., description="Folder holding original.csv and the generated full/train/val/test.csv")

    case_key: str = Field(..., description="Raw column identifying the case each event belongs to")
    activity_key: str = Field(..., description="Raw column identifying the activity label")
    resource_key: str = Field(..., description="Raw column identifying the resource")
    timestamp_key: str = Field(..., description="Raw column holding the event timestamp")
    label_key: str = Field(..., description="Raw column holding the case label")

    train_split: float = Field(..., gt=0.0, lt=1.0)
    val_split: float = Field(..., gt=0.0, lt=1.0)
    test_split: float = Field(..., gt=0.0, lt=1.0)

    max_seq_len: int = Field(
        ..., gt=0,
        description="Cases are truncated to this many events, which also bounds every prefix and suffix cut "
        "from them; the model's sequence tensors are padded to it",
    )
    
    event_features: list[str] = Field(
        ..., description="Canonical (post-preprocessing) columns the encoders read beside the "
        "activity, resource and time delta. A numeric one becomes a value and a present flag, "
        "anything else a vocabulary"
    )

    time_clip_percentile: float = Field(
        ..., gt=0.0, le=100.0,
        description="Values above this train-split percentile are clipped before normalization. "
        "Applied to every numeric channel against its own percentile: the per-event gaps the "
        "encoders read, the remaining time the model predicts, and each numeric event-feature column",
    )

    batch_size: int = Field(..., gt=0)
    num_workers: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _splits_sum_to_one(self) -> "DataConfig":
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train/val/test splits must sum to 1.0, got {total}")
        return self


class EmbeddingConfig(StrictModel):
    """Event embeddings, shared by both trace encoders and the decoder."""

    activity_dim: int = Field(..., gt=0)
    resource_dim: int = Field(..., gt=0)
    feature_dim: int = Field(
        ..., gt=0,
        description="Width of every categorical feature channel's lookup. One width for all "
        "of them, since they share a table; each channel widens the projection's input by this "
        "much, so a log with many of them wants a smaller value, not a bigger one",
    )


class TraceEncoderConfig(StrictModel):
    """Transformer encoder reading a sequence of events, every position attending over every other.

    Used twice, with its own values each time: once over the prefix, whose summary conditions
    both the prior and the decoder, and once over the ground-truth suffix, which is read on
    the training path only, to give the posterior something to encode.

    The width is absent: an encoder reads the shared event embeddings and, for the prefix, is
    read back by the decoder's cross-attention, so it runs at `ModelConfig.d_model`.
    """

    num_layers: int = Field(..., gt=0)
    num_heads: int = Field(..., gt=0, description="Attention heads per layer; must divide `d_model`")
    feedforward_dim: int = Field(..., gt=0, description="Width of the feed-forward block inside a layer")
    dropout: float = Field(..., ge=0.0, lt=1.0)


class PriorConfig(StrictModel):
    """MLP mapping the prefix summary to `p(z | prefix)`, the distribution sampled from at
    inference time.

    It takes the place a fixed `N(0, I)` takes in an unconditional VAE. Because the KL term
    compares the posterior against this conditioned prior rather than against `N(0, I)`, z
    only has to carry what the prefix does not already imply.
    """

    hidden_dims: list[int] = Field(..., description="Widths of the hidden layers; empty for a linear prior")
    dropout: float = Field(..., ge=0.0, lt=1.0)


class LatentConfig(StrictModel):
    latent_dim: int = Field(..., gt=0)


class DecoderConfig(StrictModel):
    """Transformer decoder writing the suffix: causal self-attention over the suffix so far,
    cross-attention over the encoded prefix.

    Like the encoders it runs at `ModelConfig.d_model`, which cross-attention over the prefix
    requires of it anyway.
    """

    num_layers: int = Field(..., gt=0)
    num_heads: int = Field(..., gt=0, description="Attention heads per layer; must divide `d_model`")
    feedforward_dim: int = Field(..., gt=0, description="Width of the feed-forward block inside a layer")
    dropout: float = Field(..., ge=0.0, lt=1.0)
    activity_dropout: float = Field(
        ..., ge=0.0, lt=1.0,
        description="Fraction of teacher-forced input activities blanked to PAD during "
        "training. An unreliable previous token cannot carry the suffix on its own, which "
        "pushes that information into z. 0.0 disables it",
    )
    head_hidden_dim: int = Field(..., gt=0, description="Width of the layer shared by the two output heads")


class ModelConfig(StrictModel):
    """Every hyperparameter of `TransformerCVAE`.

    Dimensions derived from the data (vocabulary sizes, special-token indices, sequence
    length) are deliberately absent: they come from `DatasetInfo` at build time, so a
    config cannot disagree with the dataset it is trained on.
    """

    d_model: int = Field(
        ..., gt=0,
        description="The one width the embeddings, both encoders and the decoder all run at. "
        "Cross-attention makes the prefix encoder and the decoder agree on it, and the shared "
        "event embeddings make the suffix encoder agree with them",
    )

    embeddings: EmbeddingConfig
    prefix_encoder: TraceEncoderConfig
    suffix_encoder: TraceEncoderConfig
    prior: PriorConfig
    latent: LatentConfig
    decoder: DecoderConfig

    @model_validator(mode="after")
    def _heads_divide_width(self) -> "ModelConfig":
        # nn.MultiheadAttention asserts this when the layer is built, halfway through a run's
        # setup. Checking it here turns a config mistake back into a config error.
        for name in ("prefix_encoder", "suffix_encoder", "decoder"):
            num_heads = getattr(self, name).num_heads
            if self.d_model % num_heads != 0:
                raise ValueError(
                    f"model.{name}.num_heads ({num_heads}) must divide model.d_model ({self.d_model})"
                )
        return self


class LossConfig(StrictModel):
    """The KL term: how it is weighted over training, and how far it is allowed to fall.

    The annealing schedule (see `training/kl.py`) is measured in optimizer steps, not
    epochs: an epoch is a different amount of learning on every log, so a schedule denominated
    in epochs has to be re-derived per dataset, while one in steps means the same thing
    everywhere.

    The cycle is given as a length rather than as a count fitted into `training.max_steps`, so
    that shortening or lengthening a run leaves the shape of the schedule alone. A budget and a
    schedule are separate decisions, and a run that stops early has still seen whole cycles.
    """

    kl_annealing_period_steps: int = Field(..., gt=0, description="Optimizer steps in one cycle")
    kl_annealing_ratio: float = Field(..., gt=0.0, lt=1.0, description="Fraction of each cycle spent ramping up")
    kl_annealing_start_weight: float = Field(..., ge=0.0, description="Weight each cycle ramps up from")
    kl_annealing_full_weight: float = Field(..., ge=0.0, description="Weight each cycle ramps up to, and holds at")

    free_bits: float = Field(
        ..., ge=0.0,
        description="Nats per latent dimension the KL is not penalized below. Unlike the "
        "annealing weight, which trades the KL off against a reconstruction sum that grows "
        "with suffix length, this is a floor on the information z carries and so means the "
        "same thing on every dataset. 0.0 leaves the KL unfloored",
    )


class OptimizerConfig(StrictModel):
    lr: float = Field(..., gt=0.0)
    weight_decay: float = Field(..., ge=0.0)


class TrainingConfig(StrictModel):
    """How long a run goes on for, and how often it looks at the validation split.

    Both are counted in optimizer steps rather than epochs. One epoch is 31 steps on sepsis
    and 863 on traffic_fines, so an epoch-denominated budget silently means something
    different on every log; a step is a step everywhere.
    """

    max_steps: int = Field(
        ..., gt=0,
        description="Ceiling on the optimizer steps a run takes. Early stopping is what "
        "normally ends a run; this is what bounds one that never plateaus",
    )
    grad_clip_norm: float | None = Field(
        None, gt=0.0, description="Max gradient norm; null or absent leaves gradients unclipped"
    )
    device: Literal["cpu", "cuda", "mps"]
    best_model_dir: Path = Field(..., description="The best step of a run, one file per run")
    log_dir: Path = Field(..., description="TensorBoard event directory")
    val_every_n_steps: int = Field(
        ..., gt=0,
        description="Steps between validations. Also the unit `early_stopping.patience` "
        "counts in, which is what makes that patience portable across datasets",
    )
    validation_pairs: int = Field(
        ..., gt=0,
        description="Pairs the two teacher-forced passes read, as a fixed slice of the "
        "validation split, or the whole of it where it is smaller. Validation is on the "
        "critical path and the whole split is more than the mean needs",
    )
    generation_pairs: int = Field(
        ..., gt=0,
        description="Prefixes the free-running pass generates for, as a fixed slice of the "
        "same split. Smaller again, a suffix costing one decoder pass per event rather than "
        "one per suffix; this is the sample the selection score is computed on",
    )


class InferenceConfig(StrictModel):
    """Generating suffixes for a whole split, which is what evaluation reads.

    The device is not repeated here: a run generates on the device it trained on
    (`training.device`).
    """

    num_samples: int = Field(
        ..., gt=0,
        description="Suffixes generated per prefix, all from that prefix's p(z | prefix); the "
        "spread across them is what the latent is claiming the prefix leaves open",
    )
    generation_rows: int = Field(
        ..., gt=0,
        description="Rows `generate` puts through the decoder in one call. A prefix reaches it "
        "`num_samples` times over, each row holding a key and value cache per position and "
        "layer, so this and not `data.batch_size` is what bounds the memory a call takes",
    )
    generations_dir: Path = Field(..., description="One generations file per run, named after it")


class EarlyStoppingConfig(StrictModel):
    """Stop training once the selection score plateaus (see `training/early_stopping.py`).

    What it watches is what best-model selection watches: the free-running generation score,
    which is comparable at every point of a run whatever the annealing weight is doing, and is
    the same quantity `pipelines/evaluate.py` reports at the end.
    """

    patience: int = Field(
        ..., gt=0,
        description="Non-improving validations tolerated before stopping. Counted in "
        "validations, so it must outlast a whole `loss.kl_annealing_ratio` ramp, during which "
        "generation gets worse by design",
    )
    min_delta_perc: float = Field(..., ge=0.0, description="Minimum relative improvement to reset the patience counter")


class ExperimentConfig(StrictModel):
    """Top-level config: the single object loaded from YAML.

    Pass sub-sections (e.g. `cfg.model.encoder`, `cfg.optimizer`) into
    functions rather than this whole object, so each function only depends
    on the parameters it actually uses.
    """

    seed: int
    experiment_name: str
    output_dir: Path

    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    early_stopping: EarlyStoppingConfig
    inference: InferenceConfig
