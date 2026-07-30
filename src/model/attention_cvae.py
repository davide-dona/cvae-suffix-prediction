from dataclasses import dataclass
from typing import Literal
import torch
from torch import nn

from src.configs.dataset_info import DatasetInfo
from src.configs.schema import ModelConfig
from src.datasets.dataset import SuffixItem
from src.model.components.decoder import Decoder, DecoderOutput, GeneratedSuffix
from src.model.components.embeddings import EventEmbeddings
from src.model.components.latent import Gaussian, PosteriorNetwork, PriorNetwork
from src.model.components.trace_encoder import TraceEncoder


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
        self.embeddings = EventEmbeddings(config=config.embeddings, dataset_info=dataset_info)
        # Same architecture, separate weights: one reads prefixes, the other ground-truth
        # suffixes, and only the second is ever run on the training path.
        self.prefix_encoder = TraceEncoder(config=config.prefix_encoder, embeddings=self.embeddings)
        self.suffix_encoder = TraceEncoder(config=config.suffix_encoder, embeddings=self.embeddings)

        # Twice the encoder's hidden size, since its two directions are concatenated.
        prefix_dim = self.prefix_encoder.output_dim
        self.prior = PriorNetwork(
            config=config.prior, latent_config=config.latent, prefix_dim=prefix_dim
        )
        self.posterior = PosteriorNetwork(
            latent_config=config.latent,
            prefix_dim=prefix_dim,
            suffix_dim=self.suffix_encoder.output_dim,
        )
        self.decoder = Decoder(
            config=config.decoder,
            attention_config=config.attention,
            latent_config=config.latent,
            embeddings=self.embeddings,
            prefix_dim=prefix_dim,
            num_activities=dataset_info.num_activities,
            num_resources=dataset_info.num_resources,
            sos_activity_index=dataset_info.sos_activity_index,
            sos_resource_index=dataset_info.sos_resource_index,
            eot_activity_index=dataset_info.eot_activity_index,
        )

        self.pad_activity_index = dataset_info.pad_activity_index
        self.pad_resource_index = dataset_info.pad_resource_index

    def forward(
        self,
        item: SuffixItem,
        *,
        sample_from: Literal['posterior', 'prior'] = 'posterior',
    ) -> AttentionCVAEOutput:
        """
        Args:
            item: A batch from `SuffixDataset`.
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
            activities=item.prefix.activities,
            resources=item.prefix.resources,
            timestamps=item.prefix.timestamps,
            lengths=item.prefix_len,
        )  # [batch_size, seq_len, prefix_dim], [batch_size, prefix_dim]

        # Always computed: it is sampled from at inference, and it is what the KL term on the
        # training path measures the posterior against.
        prior = self.prior(prefix_summary)  # mean, logvar: [batch_size, latent_dim] each

        if sample_from == 'posterior':
            # The one place the ground-truth suffix is read, and `summarize` rather than a full
            # read because the latent is a single vector: there is nothing here for per-step
            # outputs to feed, so they are never unpacked.
            suffix_summary = self.suffix_encoder.summarize(
                activities=item.suffix.activities,
                resources=item.suffix.resources,
                timestamps=item.suffix.timestamps,
                lengths=item.suffix_len,
            )  # [batch_size, suffix_dim]
            # q(z | prefix, suffix): a latent that already describes the suffix being
            # reconstructed, which is what makes the reconstruction learnable at all.
            posterior = self.posterior(
                prefix_summary=prefix_summary, suffix_summary=suffix_summary
            )
            z = posterior.sample()  # [batch_size, latent_dim]

        else:
            # Generating: the suffix is unknown, so the suffix encoder and the posterior are not
            # run and p(z | prefix) supplies the latent instead.
            posterior = None
            z = prior.sample()  # [batch_size, latent_dim]

        # The prefix reaches the decoder both ways here: summarized into its initial state, and
        # event by event as the memory its attention reads.
        decoder_output = self.decoder(
            decoder_input=item.decoder_input,
            prefix_len=item.prefix_len,
            z=z,
            prefix_outputs=prefix_outputs,
            prefix_summary=prefix_summary,
        )
        return AttentionCVAEOutput(decoder=decoder_output, prior=prior, posterior=posterior)

    @torch.no_grad()
    def generate(self, item: SuffixItem, *, num_samples: int) -> GeneratedSuffix:
        """Generate `num_samples` suffixes for every prefix in `item`.

        The suffix is unknown here, so nothing but the prefix is read: the suffix encoder and
        the posterior stay unrun and every latent comes from `p(z | prefix)`. The samples of
        one prefix differ only in that latent, since the decoder reads its heads greedily,
        which is what makes the spread across them a property of the prior rather than of a
        softmax.

        The prefix is encoded once and its outputs repeated, so the cost of a sample is the
        decoder's per-step path and nothing else. Repeating rather than looping is also what
        keeps the whole `batch_size * num_samples` in one recurrence: the per-step path is
        launch-bound, so a wider batch costs almost nothing per extra row.

        Args:
            item: A batch from `SuffixDataset`, read for its prefix only.
            num_samples: How many suffixes to draw per prefix.
        Returns:
            The generated suffixes, `[batch_size, num_samples, steps]`, with row `(i, j)` the
            j-th sample for the i-th prefix of the batch.
        """
        prefix_outputs, prefix_summary = self.prefix_encoder(
            activities=item.prefix.activities,
            resources=item.prefix.resources,
            timestamps=item.prefix.timestamps,
            lengths=item.prefix_len,
        )  # [batch_size, seq_len, prefix_dim], [batch_size, prefix_dim]

        # Every sample of a prefix gets its own row, adjacent to its siblings, so the flat
        # result reshapes straight back into [batch_size, num_samples, ...].
        prefix_outputs = prefix_outputs.repeat_interleave(repeats=num_samples, dim=0)
        prefix_summary = prefix_summary.repeat_interleave(repeats=num_samples, dim=0)
        prefix_len = item.prefix_len.repeat_interleave(repeats=num_samples, dim=0)

        # The rows of a prefix hold the same distribution, so they differ only in the draw.
        z = self.prior(prefix_summary).sample()  # [batch_size * num_samples, latent_dim]

        # A suffix holds at most `max_seq_len` events, the padded width the batch comes in at.
        generated = self.decoder.generate(
            z=z,
            prefix_outputs=prefix_outputs,
            prefix_summary=prefix_summary,
            prefix_len=prefix_len,
            max_steps=item.prefix.activities.size(dim=1),
        )
        batch_size = item.prefix_len.size(dim=0)
        return GeneratedSuffix(
            activities=generated.activities.view(batch_size, num_samples, -1),
            resources=generated.resources.view(batch_size, num_samples, -1),
            timestamps=generated.timestamps.view(batch_size, num_samples, -1),
            lengths=generated.lengths.view(batch_size, num_samples),
        )
