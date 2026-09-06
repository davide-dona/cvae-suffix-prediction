from dataclasses import dataclass
from typing import Self

import torch

from src.distributions import Gaussian
from src.scalar_metrics import ScalarMetrics


def gaussian_kl(posterior: Gaussian, prior: Gaussian) -> torch.Tensor:
    """Closed-form KL divergence between two diagonal Gaussians, per latent dimension.
    Args:
        posterior: The posterior distribution.
        prior: The prior distribution.
    Returns:
        `[batch_size, latent_dim]`, the divergence contributed by each dimension.
    """
    divergence = (
        prior.logvar
        - posterior.logvar
        + (posterior.logvar.exp() + (posterior.mean - prior.mean) ** 2) / prior.logvar.exp()
        - 1.0
    )
    return 0.5 * divergence  # [batch_size, latent_dim]


def linear_warmup_weight(step: int, *, ramp_steps: int, start: float, stop: float) -> float:
    """
    The KL weight one optimizer step is given under a linear warmup.

    The weight ramps linearly from `start` to `stop` over the first `ramp_steps` steps and holds at
    `stop` for the rest of the run, so the posterior is given one stretch of cheap latent capacity
    rather than a series of them.

    Args:
        step: The step to weight, counted from 0.
        ramp_steps: Optimizer steps spent ramping up, after which the weight is held.
        start: Weight the run ramps up from.
        stop: Weight the run ramps up to, and holds at.
    Returns:
        The weight to apply at `step`.
    """
    # Once the ramp is over the weight is held, for however long the run goes on
    if step >= ramp_steps:
        return stop
    return start + (stop - start) * (step / ramp_steps)


def free_bits_kl(kl_per_dim: torch.Tensor, *, free_bits: float) -> torch.Tensor:
    """Floor each latent dimension's KL at `free_bits` nats before it is weighted.
    This is a common trick to prevent posterior collapse, where the KL is driven to 0 and the latent
    is ignored.
    Args:
        kl_per_dim: `[batch_size, latent_dim]`, the per-dimension KL from `gaussian_kl`.
        free_bits: Nats per dimension the KL is not penalized below. 0.0 leaves it unfloored.
    Returns:
        A scalar, the batch-summed, dimension-floored KL to weight and backpropagate.
    """
    batch_size = kl_per_dim.size(0)
    return kl_per_dim.mean(dim=0).clamp(min=free_bits).sum() * batch_size


# Margin over `free_bits` a dimension has to clear to count as used: one parked on the floor pays
# no gradient and drifts around it, so counting from the floor exactly would flicker on noise.
ACTIVE_MARGIN_NATS = 0.05


@dataclass(frozen=True, slots=True)
class LatentMetrics(ScalarMetrics):
    """What z carried over one pass: the KL it holds, how many dimensions hold it, and the weight
    the KL term was charged at.

    None of the three is a term of the loss. A total KL cannot tell a collapsed latent from a few
    dead dimensions among used ones, which is what `free_bits` hides: a dimension below the floor
    is left unpenalized, so a run can pay `latent_dim * free_bits` nats and carry no information.
    `kl_weight` is only ever a constant of the step it was logged at, but is scaled with the
    other two so the same averaging recovers it. Logged rather than reported, so like `Loss` it
    declares no units.
    """

    kl_nats: float = 0.0
    active_dims: float = 0.0
    kl_weight: float = 0.0

    @classmethod
    def of(cls, kl_per_dim: torch.Tensor, *, free_bits: float, kl_weight: float) -> Self:
        """Read one batch's KL, counting dimensions on the scale `free_bits_kl` floors.

        Args:
            kl_per_dim: `[batch_size, latent_dim]`, the per-dimension KL from `gaussian_kl`.
            free_bits: Nats per dimension the KL is not penalized below.
            kl_weight: The weight this step's KL term was charged at.
        Returns:
            The metrics, scaled by the batch like the loss terms they are logged beside, so the
            caller's division by the batch recovers all three.
        """
        batch_size = kl_per_dim.size(0)
        active = (kl_per_dim.mean(dim=0) > free_bits + ACTIVE_MARGIN_NATS).sum().item()
        return cls(
            kl_nats=kl_per_dim.sum().item(),
            active_dims=active * batch_size,
            kl_weight=kl_weight * batch_size,
        )
