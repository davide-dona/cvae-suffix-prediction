import torch
from torch import nn

from src.configs.schema import LatentConfig, PriorConfig
from src.distributions.gaussian import Gaussian


class PriorNetwork(nn.Module):
    """`p(z | prefix)`: the distribution the latent is drawn from at inference time.

    It stands where the fixed `N(0, I)` stands in an unconditional VAE, its mean and variance
    produced from the prefix summary instead. Because the KL term measures the posterior
    against it, whatever the prefix already determines costs nothing to encode.
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
            prefix_summary: The prefix encoder's summary, `[batch_size, d_model]`.
        Returns:
            `p(z | prefix)`.
        """
        parameters = self.net(prefix_summary)  # [batch_size, 2 * latent_dim]
        # The head emits both halves in one tensor: first the mean, then the log-variance.
        mean, logvar = parameters.chunk(chunks=2, dim=-1)  # each [batch_size, latent_dim]
        return Gaussian.create(mean=mean, logvar=logvar)


class PosteriorNetwork(nn.Module):
    """`q(z | prefix, suffix)`: the distribution the latent is drawn from during training only.

    It hands the decoder a latent that already describes the suffix to reconstruct, which is
    what makes reconstruction learnable. At inference the suffix is unknown and the prior
    takes its place; the KL term trains the two to agree, which is what makes that
    substitution legitimate.
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
            prefix_summary: The prefix encoder's summary, `[batch_size, d_model]`.
            suffix_summary: The suffix encoder's summary, `[batch_size, d_model]`.
        Returns:
            `q(z | prefix, suffix)`.
        """
        summaries = torch.cat(
            tensors=(suffix_summary, prefix_summary), dim=-1
        )  # [batch_size, suffix + prefix]
        parameters = self.head(summaries)  # [batch_size, 2 * latent_dim]
        # The head emits both halves in one tensor: first the mean, then the log-variance.
        mean, logvar = parameters.chunk(chunks=2, dim=-1)  # each [batch_size, latent_dim]
        return Gaussian.create(mean=mean, logvar=logvar)
