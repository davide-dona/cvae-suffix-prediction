from typing import Literal

import torch
import torch.nn.functional as F

from src.datasets.dataset import SuffixItem
from src.model import TransformerCVAE
from src.training.metrics import Metrics


def real_event_mask(lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
    """A boolean mask marking the real (non-padded) positions of a padded batch.

    The opposite polarity to `trace_encoder.padding_mask`, which marks the padding instead
    because that is what `nn.Transformer` wants; both names say which way round they are.

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
    mask = real_event_mask(lengths, predicted.size(1))
    return ((predicted - target) ** 2 * mask).sum()


def gaussian_kl(
    posterior_mean: torch.Tensor,
    posterior_logvar: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_logvar: torch.Tensor,
) -> torch.Tensor:
    """Closed-form KL divergence between two diagonal Gaussians, per latent dimension.

    The prior is an argument rather than a fixed `N(0, I)` because a conditional VAE
    compares its posterior against a learned, conditioned prior: whatever the condition
    already explains then costs nothing to encode in the latent.

    The dimensions are left unsummed because `compute_loss` needs them apart: the free-bits
    floor applies to each one on its own, and a KL summed first cannot be floored afterwards.

    Args:
        posterior_mean, posterior_logvar: Parameters of `q`, `[batch_size, latent_dim]`.
        prior_mean, prior_logvar: Parameters of `p`, `[batch_size, latent_dim]`.
    Returns:
        `[batch_size, latent_dim]`, the divergence contributed by each dimension.
    """
    divergence = (
        prior_logvar
        - posterior_logvar
        + (posterior_logvar.exp() + (posterior_mean - prior_mean) ** 2) / prior_logvar.exp()
        - 1.0
    )
    return 0.5 * divergence  # [batch_size, latent_dim]


def compute_loss(
    model: TransformerCVAE,
    batch: SuffixItem,
    *,
    kl_weight: float,
    free_bits: float,
    sample_from: Literal['posterior', 'prior'] = 'posterior',
) -> tuple[torch.Tensor, Metrics]:
    """Run one training or evaluation step and report what it cost.

    The two returned values are scaled differently on purpose. The loss is divided by the
    batch size, so the gradient does not grow with it. The metrics are left as sums over the
    batch, because the caller adds them across batches and divides once by the number of
    traces seen; dividing here as well would only be undone there, and batches are not
    equal-sized.

    On the prior path there is no posterior to measure the latent against, so the KL term is
    absent and the loss is the reconstruction alone. That is what makes it comparable at every
    point of a run: it does not move when the annealing weight does.

    Args:
        model: The model to run.
        batch: A batch from `SuffixDataset`, already on the right device.
        kl_weight: The weight the KL term is given at this step (see `training/annealing.py`).
        free_bits: Nats per latent dimension the KL is not penalized below. Only the
            objective is floored; the reported KL stays the real one.
        sample_from: Which distribution z is drawn from, passed through to the model.
            Training and diagnosis use the posterior; `'prior'` scores the path inference
            actually runs on.
    Returns:
        The per-trace loss to backpropagate, and the metrics to log, summed over the batch.
    """
    output = model(batch, sample_from=sample_from)
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
    time_delta_loss = masked_mse(
        output.decoder.time_deltas, batch.suffix.time_deltas, batch.suffix_len
    )

    reconstruction_loss = activity_loss + resource_loss + time_delta_loss

    if output.posterior is None:
        # The prior path: nothing was encoded, so there is no divergence to charge for.
        kl_loss = reconstruction_loss.new_zeros(size=())
        total_loss = reconstruction_loss
    else:
        kl_per_dim = gaussian_kl(
            posterior_mean=output.posterior.mean,
            posterior_logvar=output.posterior.logvar,
            prior_mean=output.prior.mean,
            prior_logvar=output.prior.logvar,
        )  # [batch_size, latent_dim]
        # Reported as it is, so the TensorBoard curve shows how much z actually carries rather
        # than where the floor sits.
        kl_loss = kl_per_dim.sum()
        # Free bits: a dimension already spending less than `free_bits` nats is left alone, so
        # the pressure to collapse it stops at that floor. Applied to each dimension's mean
        # over the batch rather than to each sample, then rescaled back to a batch sum to match
        # how the reconstruction term is accumulated.
        penalized_kl = kl_per_dim.mean(dim=0).clamp(min=free_bits).sum() * batch_size
        total_loss = reconstruction_loss + kl_weight * penalized_kl

    metrics = Metrics(
        loss=total_loss.item(),
        reconstruction_loss=reconstruction_loss.item(),
        kl_loss=kl_loss.item(),
        activity_loss=activity_loss.item(),
        resource_loss=resource_loss.item(),
        time_delta_loss=time_delta_loss.item(),
    )
    return total_loss / batch_size, metrics
