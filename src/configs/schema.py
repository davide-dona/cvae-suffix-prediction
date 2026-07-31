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
    """Bidirectional LSTM reading a sequence of events.

    Used twice, with its own values each time: once over the prefix, whose summary conditions
    both the prior and the decoder, and once over the ground-truth suffix, which is read on
    the training path only, to give the posterior something to encode.
    """

    hidden_dim: int = Field(..., gt=0, description="Per-direction hidden size; the summary is twice this wide")
    num_layers: int = Field(..., gt=0)
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


class AttentionConfig(StrictModel):
    """Single-head scaled dot-product attention over the prefix and the suffix so far."""

    dim: int = Field(..., gt=0, description="Shared width the prefix outputs and decoder states are projected to")
    dropout: float = Field(..., ge=0.0, lt=1.0)


class DecoderConfig(StrictModel):
    hidden_dim: int = Field(..., gt=0)
    num_layers: int = Field(..., gt=0)
    dropout: float = Field(..., ge=0.0, lt=1.0)
    head_hidden_dim: int = Field(..., gt=0, description="Width of the layer shared by the three output heads")


class ModelConfig(StrictModel):
    """Every hyperparameter of `AttentionCVAE`.

    Dimensions derived from the data (vocabulary sizes, special-token indices, sequence
    length) are deliberately absent: they come from `DatasetInfo` at build time, so a
    config cannot disagree with the dataset it is trained on.
    """

    embeddings: EmbeddingConfig
    prefix_encoder: TraceEncoderConfig
    suffix_encoder: TraceEncoderConfig
    prior: PriorConfig
    latent: LatentConfig
    attention: AttentionConfig
    decoder: DecoderConfig


class LossConfig(StrictModel):
    """The KL term: how it is weighted over training, and how far it is allowed to fall.

    The annealing schedule (see `training/annealing.py`) is measured in optimizer steps, not
    epochs: an epoch is a different amount of learning on every log, so a schedule denominated
    in epochs has to be re-derived per dataset, while one in steps means the same thing
    everywhere.
    """

    kl_annealing_cycles: int = Field(..., gt=0, description="Number of cycles to fit into training")
    kl_annealing_ratio: float = Field(..., gt=0.0, le=1.0, description="Fraction of each cycle spent ramping up")
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

    What it watches is the prior-path validation loss, which carries no KL term and is
    therefore comparable at every point of a run, whatever the annealing weight is doing.
    """

    patience: int = Field(..., gt=0, description="Non-improving validations tolerated before stopping")
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
