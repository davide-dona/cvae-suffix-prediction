import math
from dataclasses import dataclass

import torch

# The latent heads emit a log-variance rather than a standard deviation, so the scale stays
# positive by construction. The range is clamped because `exp` of an unbounded head turns
# into an inf, and from there into a NaN loss, long before training would recover.
LOGVAR_MIN, LOGVAR_MAX = -10.0, 10.0


@dataclass(frozen=True)
class Gaussian:
    """A diagonal Gaussian, `[batch_size, dim]` per field."""

    mean: torch.Tensor
    logvar: torch.Tensor

    @classmethod
    def create(cls, mean: torch.Tensor, logvar: torch.Tensor) -> 'Gaussian':
        """Build a Gaussian from an already-split mean and log-variance.

        Args:
            mean: The distribution's mean.
            logvar: The raw log-variance, clamped to `[LOGVAR_MIN, LOGVAR_MAX]` before it is
                stored, so a later `exp()` cannot overflow.
        Returns:
            The Gaussian with its log-variance clamped.
        """
        return cls(mean=mean, logvar=logvar.clamp(min=LOGVAR_MIN, max=LOGVAR_MAX))

    def sample(self) -> torch.Tensor:
        """Draw one sample through the reparametrization trick, so the gradient reaches
        `mean` and `logvar`."""
        # mean + std * noise: the randomness sits in a term with no parameters, which is what
        # leaves a gradient path through `mean` and `logvar`.
        std = torch.exp(input=0.5 * self.logvar)  # [batch_size, dim]
        return self.mean + std * torch.randn_like(input=self.mean)  # [batch_size, dim]

    def nll(self, target: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of a target under this Gaussian.
        Args:
            target: The observed value, `[batch_size]`.
        Returns:
            A scalar, the sum of the negative log-likelihoods over the batch.
        """
        nll = 0.5 * (
            self.logvar + (target - self.mean) ** 2 / self.logvar.exp() + math.log(2.0 * math.pi)
        )  # [batch_size]
        return nll.sum()
