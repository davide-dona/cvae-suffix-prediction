import torch
from torch import nn

from src.configs.schema import LatentConfig, PriorConfig
from src.distributions.gaussian import Gaussian


class PriorNetwork(nn.Module):
    """`p(z | prefix)`: the distribution the latent is drawn from at inference time.

    It occupies the position the fixed `N(0, I)` occupies in an unconditional VAE. The
    difference is that its mean and variance are produced from the prefix summary rather
    than held constant. Drawing a latent is drawing from the Gaussian this particular prefix implies.

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
