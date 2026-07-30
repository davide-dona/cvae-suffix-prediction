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
    # Neither is read by the data layer or the model: the prefix is the only condition, and an
    # event is (activity, resource, time delta). They record which columns each dataset offers
    # for the two roles, and `src/notebooks/exploration.ipynb` reads them.
    condition_features: list[str] = Field(
        ..., description="Canonical (post-preprocessing) case-level columns, candidates for the CVAE condition"
    )
    attribute_features: list[str] = Field(
        ..., description="Canonical (post-preprocessing) per-event columns, candidates for reconstruction "
        "alongside activity/resource/timestamp"
    )

    time_clip_percentile: float = Field(
        ..., gt=0.0, le=100.0,
        description="Event time-deltas above this train-split percentile are clipped before normalization",
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
    head_hidden_dim: int = Field(..., gt=0, description="Width of the layer shared by the three output heads")


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
    """Parameters of the cyclical KL annealing schedule (see `training/annealing.py`)."""

    kl_annealing_cycles: int = Field(..., gt=0, description="Number of cycles to fit into training")
    kl_annealing_ratio: float = Field(..., gt=0.0, le=1.0, description="Fraction of each cycle spent ramping up")
    kl_annealing_start_weight: float = Field(..., ge=0.0, description="Weight each cycle ramps up from")
    kl_annealing_full_weight: float = Field(..., ge=0.0, description="Weight each cycle ramps up to, and holds at")


class OptimizerConfig(StrictModel):
    lr: float = Field(..., gt=0.0)
    weight_decay: float = Field(..., ge=0.0)


class TrainingConfig(StrictModel):
    max_num_epochs: int = Field(..., gt=0)
    grad_clip_norm: float | None = Field(
        None, gt=0.0, description="Max gradient norm; null or absent leaves gradients unclipped"
    )
    device: Literal["cpu", "cuda", "mps"]
    best_model_dir: Path = Field(..., description="The best epoch of a run, one file per run")
    log_dir: Path = Field(..., description="TensorBoard event directory")
    val_every_n_epochs: int = Field(..., gt=0)


class InferenceConfig(StrictModel):
    """Generating suffixes for a whole split, which is what evaluation reads.

    The device and the batch size are not repeated here: a run generates on the device it
    trained on (`training.device`) and in the batches its data section already describes
    (`data.batch_size`), noting that the decoder actually sees `batch_size * num_samples`
    rows, since every sample of a prefix is a row of its own.
    """

    num_samples: int = Field(
        ..., gt=0,
        description="Suffixes generated per prefix, all from that prefix's p(z | prefix); the "
        "spread across them is what the latent is claiming the prefix leaves open",
    )
    predictions_dir: Path = Field(..., description="One predictions file per run, named after it")


class EarlyStoppingConfig(StrictModel):
    """Stop training once the validation loss plateaus (see `training/early_stopping.py`).

    Only epochs evaluating the full KL weight are compared, since earlier
    epochs are not comparable to each other while the KL term is annealing.
    """

    patience: int = Field(..., gt=0, description="Non-improving evaluations tolerated before stopping")
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
