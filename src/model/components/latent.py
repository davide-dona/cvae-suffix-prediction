from dataclasses import dataclass

import torch
from torch import nn

from src.configs.schema import LatentConfig, PriorConfig

# The latent heads emit a log-variance rather than a standard deviation, so the scale stays
# positive by construction. The range is clamped because `exp` of an unbounded head turns
# into an inf, and from there into a NaN loss, long before training would recover.
LOGVAR_MIN, LOGVAR_MAX = -10.0, 10.0


@dataclass
class Gaussian:
    """A diagonal Gaussian over the latent space, `[batch_size, latent_dim]` per field."""
    mean: torch.Tensor
    logvar: torch.Tensor

    @classmethod
    def from_parameters(cls, parameters: torch.Tensor) -> "Gaussian":
        """Read a `[batch_size, 2 * latent_dim]` head output as a mean and a log-variance."""
        # The head emits both halves in one tensor: first the mean, then the log-variance.
        mean, logvar = parameters.chunk(chunks=2, dim=-1)  # each [batch_size, latent_dim]
        return cls(mean=mean, logvar=logvar.clamp(min=LOGVAR_MIN, max=LOGVAR_MAX))

    def sample(self) -> torch.Tensor:
        """Draw one sample through the reparametrization trick, so the gradient reaches
        `mean` and `logvar`."""
        # mean + std * noise: the randomness sits in a term with no parameters, which is what
        # leaves a gradient path through `mean` and `logvar`.
        std = torch.exp(input=0.5 * self.logvar)                        # [batch_size, latent_dim]
        return self.mean + std * torch.randn_like(input=self.mean)      # [batch_size, latent_dim]


class PriorNetwork(nn.Module):
    """`p(z | prefix)`: the distribution the latent is drawn from at inference time.

    It occupies the position the fixed `N(0, I)` occupies in an unconditional VAE. The
    difference is that its mean and variance are produced from the prefix summary rather
    than held constant, so drawing a latent is still drawing from a Gaussian, but from the
    Gaussian this particular prefix implies.

    Because the KL term measures the posterior against this distribution rather than against
    `N(0, I)`, whatever the prefix already determines costs nothing to encode, and z is left
    carrying only the variation the prefix cannot account for.
    """

    def __init__(self, config: PriorConfig, latent_config: LatentConfig, *, prefix_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        # The input is the prefix summary; each hidden layer then narrows or widens from there.
        width = prefix_dim
        for hidden_dim in config.hidden_dims:
            layers += [
                nn.Linear(in_features=width, out_features=hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=config.dropout),
            ]
            width = hidden_dim
        # The output layer emits mean and log-variance side by side, hence twice `latent_dim`.
        layers.append(
            nn.Linear(in_features=width, out_features=2 * latent_config.latent_dim)
        )

        # With `hidden_dims` empty this collapses to a single linear layer.
        self.net = nn.Sequential(*layers)

    def forward(self, prefix_summary: torch.Tensor) -> Gaussian:
        """
        Args:
            prefix_summary: The prefix encoder's summary, `[batch_size, prefix_dim]`.
        Returns:
            `p(z | prefix)`.
        """
        parameters = self.net(prefix_summary)  # [batch_size, 2 * latent_dim]
        return Gaussian.from_parameters(parameters)


class PosteriorNetwork(nn.Module):
    """`q(z | prefix, suffix)`: the distribution the latent is drawn from during training only.
    
    Its purpose is to hand the decoder a latent that already describes the suffix it is being 
    asked to reconstruct, which is what makes reconstruction learnable at all. 
    At inference the suffix is unknown, this network is not run, and the prior takes its place.

    The KL term pulls this distribution towards the prior over the course of training. That
    is what makes the substitution legitimate: the two are trained to agree, so a latent the
    prior produces can stand in for one the posterior would have produced.

    Both summaries are supplied so that the suffix is encoded relative to what the prefix
    already implies, rather than re-encoded in full.
    """

    def __init__(self, latent_config: LatentConfig, *, prefix_dim: int, suffix_dim: int):
        super().__init__()
        # Both summaries come in already encoded by a transformer each, so a single linear layer
        # is enough here; like the prior's output layer it emits mean and log-variance together.
        self.head = nn.Linear(
            in_features=suffix_dim + prefix_dim, out_features=2 * latent_config.latent_dim
        )

    def forward(self, prefix_summary: torch.Tensor, suffix_summary: torch.Tensor) -> Gaussian:
        """
        Args:
            prefix_summary: The prefix encoder's summary, `[batch_size, prefix_dim]`.
            suffix_summary: The suffix encoder's summary, `[batch_size, suffix_dim]`.
        Returns:
            `q(z | prefix, suffix)`.
        """
        summaries = torch.cat(
            tensors=(suffix_summary, prefix_summary), dim=-1
        )  # [batch_size, suffix + prefix]
        return Gaussian.from_parameters(self.head(summaries))  # head: [batch_size, 2 * latent_dim]
