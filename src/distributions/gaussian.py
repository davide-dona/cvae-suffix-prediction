from dataclasses import dataclass
import math
import torch

# The latent heads emit a log-variance rather than a standard deviation, so the scale stays
# positive by construction. The range is clamped because `exp` of an unbounded head turns
# into an inf, and from there into a NaN loss, long before training would recover.
LOGVAR_MIN, LOGVAR_MAX = -10.0, 10.0


@dataclass
class Gaussian:
    """A diagonal Gaussian, `[batch_size, dim]` per field."""
    mean: torch.Tensor
    logvar: torch.Tensor

    @classmethod
    def from_parameters(cls, parameters: torch.Tensor) -> "Gaussian":
        """Read a `[batch_size, 2 * dim]` head output as a mean and a log-variance."""
        # The head emits both halves in one tensor: first the mean, then the log-variance.
        mean, logvar = parameters.chunk(chunks=2, dim=-1)  # each [batch_size, dim]
        return cls(mean=mean, logvar=logvar.clamp(min=LOGVAR_MIN, max=LOGVAR_MAX))

    def sample(self) -> torch.Tensor:
        """Draw one sample through the reparametrization trick, so the gradient reaches
        `mean` and `logvar`."""
        # mean + std * noise: the randomness sits in a term with no parameters, which is what
        # leaves a gradient path through `mean` and `logvar`.
        std = torch.exp(input=0.5 * self.logvar)                        # [batch_size, dim]
        return self.mean + std * torch.randn_like(input=self.mean)      # [batch_size, dim]
    
    def gaussian_nll(self, target: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of a target under a predicted Gaussian distribution.
        Args:
            target: The observed value, `[batch_size]`.
        Returns:
            A scalar, the sum of the negative log-likelihoods over the batch.
        """
        nll = 0.5 * (
            self.logvar + (target - self.mean) ** 2 / self.logvar.exp() + math.log(2.0 * math.pi)
        )  # [batch_size]
        return nll.sum()
