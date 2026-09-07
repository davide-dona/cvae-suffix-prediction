from pydantic import Field

from .base import StrictModel


class DataLoaderConfig(StrictModel):
    """torch.utils.data.DataLoader parameters."""

    batch_size: int = Field(..., gt=0)
    num_workers: int = Field(..., ge=0)


class LossConfig(StrictModel):
    """KL weighting: the annealing ramp and the free-bits floor.

    The ramp is a number of optimizer steps rather than a fraction of `training.max_steps`, so it
    means the same thing on every dataset and a run that stops early has still spent the same
    time reaching full weight.
    """

    kl_annealing_ramp_steps: int = Field(..., gt=0)
    kl_annealing_start_weight: float = Field(..., ge=0.0)
    kl_annealing_full_weight: float = Field(..., ge=0.0)
    free_bits: float = Field(
        ..., ge=0.0, description='Nats per latent dimension the KL is not penalized below'
    )


class OptimizerConfig(StrictModel):
    """Adam and the ramp its learning rate reaches `lr` over.

    The ramp is counted in optimizer steps rather than epochs, like everything else in
    `TrainingConfig`, so it means the same thing on every dataset. Nothing decays it afterwards:
    a decay spanning `training.max_steps` would never be reached, early stopping ending a run
    long before that, which would leave the schedule's shape a property of where a run happened
    to stop rather than of the config.
    """

    lr: float = Field(..., gt=0.0, description='The peak, reached at the end of the warmup')
    weight_decay: float = Field(..., ge=0.0)
    warmup_steps: int = Field(
        ...,
        ge=0,
        description='Steps the learning rate is ramped linearly to `lr` over, from one step of '
        'it. A wide batch takes a few hundred steps to settle, and starting it at full rate is '
        'what makes those steps the ones a run never recovers from. 0 starts at `lr`',
    )


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
        None, gt=0.0, description='Null or absent leaves gradients unclipped'
    )
    device: str = Field(..., pattern=r'^(cpu|mps|cuda(:\d+)?)$')
    val_every_n_steps: int = Field(
        ..., gt=0, description='Also the unit `early_stopping.patience` counts in'
    )
    validation_pairs: int = Field(
        ..., gt=0, description='Teacher-forced pairs read per validation, capped by split size'
    )
    generation_pairs: int = Field(
        ...,
        gt=0,
        description='Free-running prefixes per validation; the sample the selection score is '
        'computed on',
    )


class EarlyStoppingConfig(StrictModel):
    """Stops training once the free-running generation score plateaus, the same score used for
    best-model selection.
    """

    patience: int = Field(
        ..., gt=0, description='Non-improving validations tolerated before stopping'
    )
    min_delta_perc: float = Field(
        ..., ge=0.0, description='Minimum relative improvement to reset the patience counter'
    )
