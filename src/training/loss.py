import torch
import torch.nn.functional as F

from src.datasets.dataset import SuffixItem
from src.model import TransformerCVAE
from src.training.metrics import Metrics


def length_mask(lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
    """A boolean mask marking the real (non-padded) positions of a padded batch.

    Args:
        lengths: Real length per sequence, `[batch_size]`.
        seq_len: The padded length.
    Returns:
        `[batch_size, seq_len]`, True where the position holds a real event.
    """
    positions = torch.arange(seq_len, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def masked_mse(predicted: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Summed squared error over the real positions of a padded batch.

    Padding is excluded rather than zeroed, so how much a batch happens to be padded
    cannot influence the gradient.

    Args:
        predicted: `[batch_size, seq_len]`.
        target: `[batch_size, seq_len]`.
        lengths: Real length per sequence, `[batch_size]`.
    Returns:
        A scalar.
    """
    mask = length_mask(lengths, predicted.size(1))
    return ((predicted - target) ** 2 * mask).sum()


def gaussian_kl(
    posterior_mean: torch.Tensor,
    posterior_logvar: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_logvar: torch.Tensor,
) -> torch.Tensor:
    """Closed-form KL divergence between two diagonal Gaussians, summed over the batch.

    The prior is an argument rather than a fixed `N(0, I)` because a conditional VAE
    compares its posterior against a learned, conditioned prior: whatever the condition
    already explains then costs nothing to encode in the latent.

    Args:
        posterior_mean, posterior_logvar: Parameters of `q`, `[batch_size, latent_dim]`.
        prior_mean, prior_logvar: Parameters of `p`, `[batch_size, latent_dim]`.
    Returns:
        A scalar.
    """
    divergence = (
        prior_logvar
        - posterior_logvar
        + (posterior_logvar.exp() + (posterior_mean - prior_mean) ** 2) / prior_logvar.exp()
        - 1.0
    )
    return 0.5 * divergence.sum()


def compute_loss(
    model: TransformerCVAE, batch: SuffixItem, kl_weight: float
) -> tuple[torch.Tensor, Metrics]:
    """Run one training or evaluation step and report what it cost.

    The two returned values are scaled differently on purpose. The loss is divided by the
    batch size, so the gradient does not grow with it. The metrics are left as sums over the
    batch, because `run_epoch` adds them across batches and divides once by the size of the
    split; dividing here as well would only be undone there, and batches are not equal-sized.

    Args:
        model: The model to run.
        batch: A batch from `SuffixDataset`, already on the right device.
        kl_weight: The weight the KL term is given this epoch (see `training/annealing.py`).
    Returns:
        The per-trace loss to backpropagate, and the metrics to log, summed over the batch.
    """
    output = model(batch)
    batch_size = batch.suffix.activities.size(0)

    # `ignore_index` drops the padded suffix positions, so predictions past the end of a
    # trace neither contribute a gradient nor dilute the reported loss.
    activity_loss = F.cross_entropy(
        output.decoder.activity_logits.transpose(1, 2),
        batch.suffix.activities,
        ignore_index=model.pad_activity_index,
        reduction='sum',
    )
    resource_loss = F.cross_entropy(
        output.decoder.resource_logits.transpose(1, 2),
        batch.suffix.resources,
        ignore_index=model.pad_resource_index,
        reduction='sum',
    )
    timestamp_loss = masked_mse(
        output.decoder.timestamps, batch.suffix.timestamps, batch.suffix_len
    )

    reconstruction_loss = activity_loss + resource_loss + timestamp_loss
    kl_loss = gaussian_kl(
        output.posterior.mean, output.posterior.logvar, output.prior.mean, output.prior.logvar
    )
    total_loss = reconstruction_loss + kl_weight * kl_loss

    metrics = Metrics(
        loss=total_loss.item(),
        reconstruction_loss=reconstruction_loss.item(),
        kl_loss=kl_loss.item(),
        activity_loss=activity_loss.item(),
        resource_loss=resource_loss.item(),
        timestamp_loss=timestamp_loss.item(),
    )
    return total_loss / batch_size, metrics
