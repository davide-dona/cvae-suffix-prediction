from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .base import NAME_PATTERN, StrictModel
from .training import LossConfig


class EmbeddingConfig(StrictModel):
    """Event embeddings, shared by the trace encoder and the decoder."""

    activity_dim: int = Field(..., gt=0)
    resource_dim: int = Field(..., gt=0)
    feature_dim: int = Field(
        ..., gt=0, description='Width of every categorical event feature table, shared across them'
    )


class TraceEncoderConfig(StrictModel):
    """Transformer encoder over a sequence of events, full self-attention.

    One stack reads both sequences: the prefix, whose summary conditions the prior and whose
    events the decoder cross-attends over, and, on the training path only, the ground-truth
    suffix, whose summary feeds the posterior. A model with no latent reads the prefix alone.
    """

    num_layers: int = Field(..., gt=0)
    num_heads: int = Field(..., gt=0)
    feedforward_dim: int = Field(..., gt=0)
    dropout: float = Field(..., ge=0.0, lt=1.0)


class PriorConfig(StrictModel):
    """MLP mapping the prefix summary to p(z | prefix)."""

    hidden_dims: list[int] = Field(..., description='Empty for a linear prior')
    dropout: float = Field(..., ge=0.0, lt=1.0)


class LatentConfig(StrictModel):
    latent_dim: int = Field(..., gt=0)


class SamplingConfig(StrictModel):
    """How an architecture that draws from its activity head shapes those logits before reading
    them.

    Inference-time only, and chosen after training rather than with it: nothing here reaches a
    gradient. `temperature: 1.0, top_p: 1.0` is the untouched softmax, which is what a run trains
    and selects its checkpoint under; `pipelines.tune` searches the pair on the validation split
    and `pipelines.generate` is handed what it picked.
    """

    temperature: float = Field(
        ...,
        gt=0.0,
        description='Divides the logits before the softmax. Above 1 flattens the head, below 1 '
        'sharpens it towards its mode, and every step is scaled alike, the ones the process '
        'already determines included',
    )
    top_p: float = Field(
        ...,
        gt=0.0,
        le=1.0,
        description='Nucleus: the smallest set of activities whose probability sums to this is '
        'kept and the rest are dropped. Unlike the temperature it reads how peaked the step is, '
        'so a step with one plausible activity keeps one candidate and emits no noise while a '
        'genuine choice point keeps the activities that make it one. 1.0 drops nothing',
    )


class DecoderConfig(StrictModel):
    """Transformer decoder writing the suffix: causal self-attention plus cross-attention over
    the encoded prefix.
    """

    num_layers: int = Field(..., gt=0)
    num_heads: int = Field(..., gt=0)
    feedforward_dim: int = Field(..., gt=0)
    dropout: float = Field(..., ge=0.0, lt=1.0)
    activity_dropout: float = Field(
        ...,
        ge=0.0,
        lt=1.0,
        description='Fraction of teacher-forced activities blanked to PAD during training, so '
        'the decoder cannot answer a position from the token before it alone. What that pushes '
        'it onto depends on the architecture - z where there is one, the cross-attended prefix '
        'and the position otherwise - but the cost is the same either way, which is why both '
        'arms carry it',
    )
    head_hidden_dim: int = Field(
        ..., gt=0, description='Width of the layer shared by the two output heads'
    )


class BackboneConfig(StrictModel):
    """What every architecture here is built from: one embedding space, one encoder stack, one
    decoder stack.

    Data-derived dimensions (vocabulary sizes, special-token indices, sequence length) are
    absent: they come from `DatasetCodec` at build time.
    """

    name: str = Field(..., pattern=NAME_PATTERN)
    d_model: int = Field(
        ..., gt=0, description='Shared width for the embeddings, the encoder and the decoder'
    )

    embeddings: EmbeddingConfig
    encoder: TraceEncoderConfig
    decoder: DecoderConfig

    @model_validator(mode='after')
    def _heads_divide_width(self) -> BackboneConfig:
        # nn.MultiheadAttention asserts this when the layer is built, halfway through a run's
        # setup. Checking it here turns a config mistake back into a config error.
        for name in ('encoder', 'decoder'):
            num_heads = getattr(self, name).num_heads
            if self.d_model % num_heads != 0:
                raise ValueError(
                    f'model.{name}.num_heads ({num_heads}) must divide '
                    f'model.d_model ({self.d_model})'
                )
        return self


class CVAEConfig(BackboneConfig):
    """Every hyperparameter of `TransformerCVAE`: the backbone, the latent path the variability
    lives in, and the loss that latent is trained against - the only architecture here with a KL
    term to weigh."""

    kind: Literal['cvae']

    prior: PriorConfig
    latent: LatentConfig
    loss: LossConfig


class TransformerConfig(BackboneConfig):
    """Every hyperparameter of `Transformer`: the backbone, and how its activity head is read.

    It carries no `prior` or `latent` section, and `extra='forbid'` is what turns writing one
    into a config error rather than a silently ignored block. `sampling` is here rather than
    beside the sample counts in `InferenceConfig` for the same reason in the other direction: it
    means nothing to an architecture whose decode is greedy, so writing it under a `cvae` config
    is a config error rather than a block that reads as if it did something.
    """

    kind: Literal['transformer']

    sampling: SamplingConfig


# Which architecture a run builds is read off `model.kind`, so a config names its own class rather
# than the pipeline guessing from the sections it happens to carry.
ModelConfig = Annotated[CVAEConfig | TransformerConfig, Field(discriminator='kind')]


def sampling_of(model: ModelConfig) -> SamplingConfig | None:
    """The sampler an architecture declares, or None for one whose decode is greedy.

    Written once here rather than at each caller, so the question 'does this architecture shape
    its logits' is answered by the union itself and a third architecture answers it by being
    added to the union.

    Args:
        model: The `model` section of a validated config.
    Returns:
        Its `sampling`, or None where the architecture has no such section.
    """
    return model.sampling if isinstance(model, TransformerConfig) else None
