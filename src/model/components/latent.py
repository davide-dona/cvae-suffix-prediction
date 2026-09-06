import torch
from torch import nn

from src.configs.schema import LatentConfig, PriorConfig
from src.distributions import Gaussian


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
        layers.append(nn.Linear(in_features=width, out_features=2 * latent_config.latent_dim))

        # With `hidden_dims` empty this collapses to a single linear layer.
        self.net = nn.Sequential(*layers)

    def forward(self, prefix_summary: torch.Tensor) -> Gaussian:
        """
        Args:
            prefix_summary: The encoder's summary of the prefix, `[batch_size, d_model]`.
        Returns:
            `p(z | prefix)`.
        """
        return Gaussian.from_head(self.net(prefix_summary))  # [batch_size, latent_dim] per field


class PosteriorNetwork(nn.Module):
    """`q(z | prefix, suffix)`: the distribution the latent is drawn from during training only.

    It hands the decoder a latent that already describes the suffix to reconstruct, which is
    what makes reconstruction learnable. At inference the suffix is unknown and the prior
    takes its place; the KL term trains the two to agree, which is what makes that
    substitution legitimate.
    """

    def __init__(self, latent_config: LatentConfig, *, summary_dim: int):
        super().__init__()
        # Both summaries come out of the same encoder, so a single linear layer is enough here;
        # like the prior's output layer it emits mean and log-variance together.
        self.head = nn.Linear(
            in_features=2 * summary_dim, out_features=2 * latent_config.latent_dim
        )

    def forward(self, prefix_summary: torch.Tensor, suffix_summary: torch.Tensor) -> Gaussian:
        """
        Args:
            prefix_summary: The encoder's summary of the prefix, `[batch_size, d_model]`.
            suffix_summary: The encoder's summary of the suffix, `[batch_size, d_model]`.
        Returns:
            `q(z | prefix, suffix)`.
        """
        summaries = torch.cat(
            tensors=(suffix_summary, prefix_summary), dim=-1
        )  # [batch_size, suffix + prefix]
        return Gaussian.from_head(self.head(summaries))  # [batch_size, latent_dim] per field
