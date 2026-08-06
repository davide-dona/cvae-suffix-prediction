from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.datasets.dataset import SplitTrace
from src.metrics import ScalarMetrics
from src.model import TransformerCVAEOutput
from src.training.kl import free_bits_kl, gaussian_kl


@dataclass(frozen=True)
class Loss(ScalarMetrics):
    """The loss of one pass, and the terms it is made of."""

    loss: float = 0.0
    reconstruction_loss: float = 0.0
    kl_loss: float = 0.0
    penalized_kl_loss: float = 0.0
    activity_loss: float = 0.0
    remaining_time_loss: float = 0.0


def compute_loss(
    output: TransformerCVAEOutput,
    batch: SplitTrace,
    *,
    pad_activity_index: int,
    kl_weight: float,
    free_bits: float,
) -> tuple[torch.Tensor, Loss]:
    """Score a forward pass's predictions against the batch it was run on.
    Args:
        output: The model's prediction for `batch`, from `model(batch)`.
        batch: A batch from `TraceDataset`, already on the right device.
        pad_activity_index: `model.pad_activity_index`, ignored in the activity cross-entropy.
        kl_weight: The weight the KL term is given at this step (see `training/kl.py`).
        free_bits: Nats per latent dimension the KL is not penalized below (see `free_bits_kl`).
    Returns:
        The per-trace loss to backpropagate, and the metrics to log, summed over the batch.
    """
    batch_size = batch.suffix.activities.size(0)

    # Compute activity and remaining time reconstruction losses, summed over the batch.
    activity_loss = F.cross_entropy(
        output.decoder.activity_logits.transpose(1, 2),
        batch.suffix.activities,
        ignore_index=pad_activity_index,
        reduction='sum',
    )
    remaining_time_loss = output.decoder.remaining_time_distr.nll(batch.remaining_time)
    reconstruction_loss = activity_loss + remaining_time_loss

    # Compute KL divergence, summed over the batch
    kl_per_dim = gaussian_kl(
        posterior=output.posterior, prior=output.prior
    )  # [batch_size, latent_dim]
    kl_loss = kl_per_dim.sum()
    penalized_kl_loss = free_bits_kl(kl_per_dim, free_bits=free_bits)
    total_loss = reconstruction_loss + kl_weight * penalized_kl_loss

    metrics = Loss(
        loss=total_loss.item(),
        reconstruction_loss=reconstruction_loss.item(),
        kl_loss=kl_loss.item(),
        penalized_kl_loss=penalized_kl_loss.item(),
        activity_loss=activity_loss.item(),
        remaining_time_loss=remaining_time_loss.item(),
    )
    return total_loss / batch_size, metrics
