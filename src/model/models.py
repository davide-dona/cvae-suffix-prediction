from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from pydantic import TypeAdapter
from torch import nn

from src.configs.schema import CVAEConfig, ModelConfig
from src.datasets.codec import DatasetCodec
from src.datasets.dataset import SplitTrace
from src.distributions import Gaussian
from src.model.checkpoint import MODEL_KEYS, require_keys
from src.model.components.decoder import DecoderOutput, GeneratedSuffix
from src.training import LatentMetrics, Loss

# `ModelConfig` is a tagged union rather than a class, so a checkpoint's stored config is
# validated through an adapter rather than by calling `model_validate` on it.
_MODEL_CONFIG = TypeAdapter(ModelConfig)


@dataclass(frozen=True)
class Latents:
    """The two distributions the KL term compares, produced only by a model that has a latent."""

    prior: Gaussian  # p(z | prefix)
    posterior: Gaussian  # q(z | prefix, suffix)


@dataclass(frozen=True)
class ModelOutput:
    """What one training pass produced, whichever architecture ran it.

    `latents` is the whole of the difference the loss sees: with them the pass is scored by the
    ELBO, without them by its reconstruction alone.
    """

    decoder: DecoderOutput
    latents: Latents | None  # None for a model with no latent


def _timed_positions(batch: SplitTrace) -> torch.Tensor:
    """Mark the suffix positions the time targets are defined at.

    Args:
        batch: A batch from `TraceDataset`, whose `suffix.length` counts the EOT closing it.
    Returns:
        `[batch_size, seq_len]`, True at the positions holding a real event.
    """
    positions = torch.arange(
        end=batch.suffix.activities.size(dim=1), device=batch.suffix.length.device
    )  # [seq_len]
    return positions.unsqueeze(dim=0) < (batch.suffix.length - 1).unsqueeze(dim=1)


def time_mae(prediction: torch.Tensor, target: torch.Tensor, batch: SplitTrace) -> torch.Tensor:
    """Score one time head over the positions its target is defined at.

    Shared by every architecture, and the same call for both: neither has an opinion about where a
    time target is scored or how. The absolute error rather than a squared one because its
    minimizer is the conditional *median*, which is what the report's `*_ae_*_days` columns score.
    A squared error would fit the conditional mean instead, over durations that are standardized
    raw minutes with nothing but a 99.9-percentile filter guarding their tail, and the tail would
    set the fit.

    Args:
        prediction: The head's output, `[batch_size, seq_len]`, standardized.
        target: What it is scored against, shaped and scaled like `prediction`.
        batch: The batch the scored positions are read off.
    Returns:
        A scalar, summed over the scored positions of the whole batch.
    """
    error = (prediction - target).abs()  # [batch_size, seq_len]
    return error.masked_fill(mask=~_timed_positions(batch), value=0.0).sum()


class SuffixModel(nn.Module, ABC):
    """What the training loop, the validation pass and the generation pipeline ask of a model.

    Deliberately narrow: one pass, one generation, one loss, and the padding index the loss
    ignores. Everything that separates the architectures - where the variability lives, how the
    heads are read, what the loss charges - is answered behind this rather than branched on in
    front of it.
    """

    def __init__(self, codec: DatasetCodec):
        super().__init__()
        # Read off the codec rather than passed down: which index means padding is a property of
        # the dataset, and every model built against one agrees about it.
        self.pad_activity_index = codec.activity.pad_index
        self.eot_activity_index = codec.activity.eot_index

    @abstractmethod
    def forward(self, item: SplitTrace) -> ModelOutput:
        """Score one batch teacher-forced, for the loss to charge."""

    @abstractmethod
    def generate(
        self, item: SplitTrace, *, num_samples: int, sample: bool = True
    ) -> GeneratedSuffix:
        """Write `num_samples` suffixes for every prefix of a batch.

        Args:
            item: A batch from `TraceDataset`, read for its prefix only.
            num_samples: How many suffixes to draw per prefix.
            sample: Whether this is a draw rather than the model's single point prediction. What
                is drawn from is the architecture's business.
        Returns:
            The suffixes, `[batch_size, num_samples, ...]`.
        """

    @abstractmethod
    def compute_loss(
        self, output: ModelOutput, batch: SplitTrace, *, step: int
    ) -> tuple[torch.Tensor, Loss, LatentMetrics | None]:
        """Score a forward pass against the batch it was run on, ready to backpropagate.

        Args:
            output: This model's prediction for `batch`, from `self(batch)`.
            batch: A batch from `TraceDataset`, already on the right device.
            step: The optimizer step this pass belongs to, for a model whose loss anneals a term
                over the run. A model with nothing to anneal ignores it.
        Returns:
            The per-trace loss to backpropagate, the terms it is made of, and what the latent
            carried, both summed over the batch. The last is `None` where the model has no latent
            for the watchdogs to read.
        """

    def _per_sample(self, generated: GeneratedSuffix, *, batch_size: int) -> GeneratedSuffix:
        """Split a decoder's flat rows back into the prefix each belongs to.

        Args:
            generated: What `Decoder.generate` wrote for `batch_size * num_samples` rows, the
                samples of one prefix adjacent, as `repeat_interleave` laid them out.
            batch_size: How many prefixes those rows came from.
        Returns:
            The same suffixes as `[batch_size, num_samples, ...]`.
        """
        return GeneratedSuffix(
            activities=generated.activities.view(batch_size, -1, generated.activities.size(dim=1)),
            lengths=generated.lengths.view(batch_size, -1),
            cycle_times=generated.cycle_times.view(
                batch_size, -1, generated.cycle_times.size(dim=1)
            ),
            remaining_time=generated.remaining_time.view(batch_size, -1),
        )


# Imported after `SuffixModel` is defined: both modules import it back, so the base class has to
# already be bound in this module's namespace by the time they run.
from src.model.architectures.cvae import TransformerCVAE  # noqa: E402
from src.model.architectures.transformer import Transformer  # noqa: E402


def build_model(config: ModelConfig, codec: DatasetCodec) -> SuffixModel:
    """Build the architecture a config declares.

    Args:
        config: The run's `model` section, whose `kind` names the architecture.
        codec: The dataset the model is built against, supplying every data-derived dimension.
    Returns:
        The model, on the CPU and in training mode.
    """
    if isinstance(config, CVAEConfig):
        return TransformerCVAE(config=config, codec=codec)
    return Transformer(config=config, codec=codec)


def model_from_checkpoint(
    checkpoint: dict, codec: DatasetCodec, *, device: str = 'cpu'
) -> SuffixModel:
    """Rebuild the model a checkpoint holds, with its weights loaded.

    Which architecture that is comes out of the checkpoint's own config, so a checkpoint is read
    without being told what wrote it.

    Args:
        checkpoint: A checkpoint read by `load_checkpoint`.
        codec: The dataset the model is to be used on, supplying the vocabulary
            sizes and sequence length it was built against.
        device: Where to place the model.
    Returns:
        The model, in evaluation mode.
    Raises:
        ValueError: If the checkpoint does not carry a config and weights.
        pydantic.ValidationError: If the config it carries names no architecture, which is what
            a checkpoint written before `model.kind` existed looks like from here.
    """
    require_keys(checkpoint, MODEL_KEYS, purpose='rebuilt', remedy='Train the model again.')
    config = _MODEL_CONFIG.validate_python(checkpoint['model_config'])

    model = build_model(config=config, codec=codec).to(device=device)
    model.load_state_dict(state_dict=checkpoint['model_state_dict'])
    model.eval()
    return model
