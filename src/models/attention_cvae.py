from dataclasses import dataclass
from typing import Literal
import torch
from torch import nn

from src.configs.dataset_info import DatasetInfo
from src.configs.schema import ModelConfig
from src.models.components.decoder import Decoder, DecoderOutput
from src.models.components.embeddings import EventEmbeddings
from src.models.components.latent import Gaussian, PosteriorNetwork, PriorNetwork
from src.models.components.trace_encoder import TraceEncoder


@dataclass
class AttentionCVAEOutput:
    decoder: DecoderOutput
    prior: Gaussian
    # None when sampling from the prior, since the suffix is then not read at all.
    posterior: Gaussian | None


class AttentionCVAE(nn.Module):
    """
    A conditional VAE that predicts a trace's suffix from its prefix.

    The prefix is the condition, and it reaches the decoder two ways: 
    - summarized, through the initial state;
    - event by event, through attention. 
    
    Because the decoder can read the prefix directly and the prior is 
    conditioned on it too, the latent z is left encoding only what the prefix 
    does not determine.

    Flow, with `prefix` and `suffix` both padded to `max_seq_len`:
        prefix              -> prefix_outputs, prefix_summary
        prefix_summary      -> p(z | prefix)
        + suffix_summary    -> q(z | prefix, suffix)      (training only)
        z ~ p(z | prefix)                                 (q(z | prefix, suffix) during training)
        z, prefix_*, suffix -> activity/resource/timestamp predictions
    """

    def __init__(self, config: ModelConfig, dataset_info: DatasetInfo):
        super().__init__()

        # Built once and handed to every part below, so an activity means the same vector
        # whether it is being read in a prefix or written into a suffix.
        self.embeddings = EventEmbeddings(config.embeddings, dataset_info)
        # Same architecture, separate weights: one reads prefixes, the other ground-truth
        # suffixes, and only the second is ever run on the training path.
        self.prefix_encoder = TraceEncoder(config.prefix_encoder, self.embeddings)
        self.suffix_encoder = TraceEncoder(config.suffix_encoder, self.embeddings)

        # Twice the encoder's hidden size, since its two directions are concatenated.
        prefix_dim = self.prefix_encoder.output_dim
        self.prior = PriorNetwork(config.prior, config.latent, prefix_dim=prefix_dim)
        self.posterior = PosteriorNetwork(
            config.latent, prefix_dim=prefix_dim, suffix_dim=self.suffix_encoder.output_dim
        )
        self.decoder = Decoder(
            config.decoder,
            config.attention,
            config.latent,
            self.embeddings,
            prefix_dim=prefix_dim,
            num_activities=dataset_info.num_activities,
            num_resources=dataset_info.num_resources,
        )

        self.pad_activity_index = dataset_info.pad_activity_index
        self.pad_resource_index = dataset_info.pad_resource_index

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        sample_from: Literal['posterior', 'prior'] = 'posterior',
    ) -> AttentionCVAEOutput:
        """
        Args:
            batch: A batch from `SuffixDataset`.
            sample_from: Which distribution z is drawn from. Training uses the posterior,
                the only path on which the ground-truth suffix is read at all. Generating a
                suffix for an unseen prefix uses the prior, which the KL term has spent
                training pulling the posterior towards.
        Returns:
            The decoder's predictions and the latent distributions the loss compares.
        """
        # The prefix is read once, in two forms: event by event for attention to look at, and
        # as one summary for the latent networks and the decoder's initial state.
        prefix_outputs, prefix_summary = self.prefix_encoder(
            batch['prefix_activities'],
            batch['prefix_resources'],
            batch['prefix_timestamps'],
            batch['prefix_len'],
        )  # [batch_size, seq_len, prefix_dim], [batch_size, prefix_dim]

        # Always computed: it is sampled from at inference, and it is what the KL term on the
        # training path measures the posterior against.
        prior = self.prior(prefix_summary)  # mean, logvar: [batch_size, latent_dim] each

        if sample_from == 'posterior':
            # The one place the ground-truth suffix is read. Its per-step outputs are dropped:
            # only the summary is wanted, since the latent is a single vector.
            _, suffix_summary = self.suffix_encoder(
                batch['target_activities'],
                batch['target_resources'],
                batch['target_timestamps'],
                batch['suffix_len'],
            )  # [batch_size, suffix_dim]
            # q(z | prefix, suffix): a latent that already describes the suffix being
            # reconstructed, which is what makes the reconstruction learnable at all.
            posterior = self.posterior(prefix_summary, suffix_summary)
            z = posterior.sample()  # [batch_size, latent_dim]

        else:
            # Generating: the suffix is unknown, so the suffix encoder and the posterior are not
            # run and p(z | prefix) supplies the latent instead.
            posterior = None
            z = prior.sample()  # [batch_size, latent_dim]

        # The prefix reaches the decoder both ways here: summarized into its initial state, and
        # event by event as the memory its attention reads.
        decoder_output = self.decoder(batch, z, prefix_outputs, prefix_summary)
        return AttentionCVAEOutput(decoder=decoder_output, prior=prior, posterior=posterior)
