from __future__ import annotations
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base for all config sections: immutable and typo-proof."""

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

    max_seq_len: int = Field(..., gt=0, description="Max prefix/suffix length in events; longer traces are truncated")
    vocab_size: int = Field(..., gt=0, description="Number of distinct activity labels")
    condition_features: list[str] = Field(
        ..., description="Canonical (post-preprocessing) case-level columns used as the CVAE condition"
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


class EncoderConfig(StrictModel):
    embedding_dim: int = Field(..., gt=0)
    hidden_dim: int = Field(..., gt=0)
    num_layers: int = Field(..., gt=0)
    dropout: float = Field(..., ge=0.0, lt=1.0)


class DecoderConfig(StrictModel):
    hidden_dim: int = Field(..., gt=0)
    num_layers: int = Field(..., gt=0)
    dropout: float = Field(..., ge=0.0, lt=1.0)


class LatentConfig(StrictModel):
    latent_dim: int = Field(..., gt=0)
    condition_dim: int = Field(..., gt=0)


class ModelConfig(StrictModel):
    encoder: EncoderConfig
    decoder: DecoderConfig
    latent: LatentConfig


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
    grad_clip_norm: float = Field(..., gt=0.0)
    device: Literal["cpu", "cuda", "mps"]
    checkpoint_dir: Path
    log_dir: Path = Field(..., description="TensorBoard event directory")
    log_every_n_steps: int = Field(..., gt=0)
    val_every_n_epochs: int = Field(..., gt=0)


class EarlyStoppingConfig(StrictModel):
    """Stop training once the validation loss plateaus (see `training/early_stopping.py`).

    Only epochs evaluating the full KL weight are compared, since earlier
    epochs are not comparable to each other while the KL term is annealing.
    """

    patience: int = Field(..., gt=0, description="Non-improving evaluations tolerated before stopping")
    min_delta_perc: float = Field(..., ge=0.0, description="Minimum relative improvement to reset the patience counter")
    debug: bool = Field(..., description="Write per-epoch stopper state to a debug file")


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
