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
from src.model.components.trace_encoder import TraceEncoder, padding_mask


@dataclass
class TransformerCVAEOutput:
    decoder: DecoderOutput
    prior: Gaussian
    # None when sampling from the prior, since the suffix is then not read at all.
    posterior: Gaussian | None


class TransformerCVAE(nn.Module):
    """
    A conditional VAE that predicts a trace's suffix from its prefix.

    The prefix is the condition, and the decoder cross-attends over it at every position and
    in every layer: over each of its events, and over the CLS row summarizing the whole of it.
    Because the decoder can read the prefix directly and the prior is conditioned on it too, the
    latent z is left encoding only what the prefix does not determine.

    Flow, with `prefix` and `suffix` both padded to `max_seq_len`:
        prefix              -> prefix_encoded (a row per event, plus a CLS summary row)
        prefix summary      -> p(z | prefix)
        + suffix summary    -> q(z | prefix, suffix)      (training only)
        z ~ p(z | prefix)                                 (q(z | prefix, suffix) during training)
        z, prefix_encoded, suffix -> activity/resource/timestamp predictions
    """

    def __init__(self, config: ModelConfig, dataset_info: DatasetInfo):
        super().__init__()

        # Built once and handed to every part below, so an activity means the same vector
        # whether it is being read in a prefix or written into a suffix.
        self.embeddings = EventEmbeddings(
            config=config.embeddings, dataset_info=dataset_info, d_model=config.d_model
        )
        # Same architecture, separate weights: one reads prefixes, the other ground-truth
        # suffixes, and only the second is ever run on the training path.
        self.prefix_encoder = TraceEncoder(
            config=config.prefix_encoder, embeddings=self.embeddings, d_model=config.d_model
        )
        self.suffix_encoder = TraceEncoder(
            config=config.suffix_encoder, embeddings=self.embeddings, d_model=config.d_model
        )

        # Both summaries are CLS rows off an encoder, so both come in at `d_model`.
        self.prior = PriorNetwork(
            config=config.prior, latent_config=config.latent, prefix_dim=config.d_model
        )
        self.posterior = PosteriorNetwork(
            latent_config=config.latent, prefix_dim=config.d_model, suffix_dim=config.d_model
        )
        self.decoder = Decoder(
            config=config.decoder,
            latent_config=config.latent,
            embeddings=self.embeddings,
            d_model=config.d_model,
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
    ) -> TransformerCVAEOutput:
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
        # Built once and read twice, by the encoder and by the decoder's cross-attention, so the
        # two cannot disagree about which rows of the encoded prefix hold real events.
        prefix_pad_mask = padding_mask(
            lengths=item.prefix_len, seq_len=item.prefix.activities.size(dim=1)
        )  # [batch_size, 1 + seq_len]

        # The prefix is read once, into a form serving both purposes: a row per event for the
        # decoder to attend over, and the CLS row the latent networks summarize it by.
        prefix_encoded = self.prefix_encoder(
            activities=item.prefix.activities,
            resources=item.prefix.resources,
            timestamps=item.prefix.timestamps,
            pad_mask=prefix_pad_mask,
        )  # [batch_size, 1 + seq_len, d_model]
        prefix_summary = prefix_encoded[:, 0]  # [batch_size, d_model], the CLS row

        # Always computed: it is sampled from at inference, and it is what the KL term on the
        # training path measures the posterior against.
        prior = self.prior(prefix_summary)  # mean, logvar: [batch_size, latent_dim] each

        if sample_from == 'posterior':
            # The one place the ground-truth suffix is read, and only for its summary: the
            # latent is a single vector, so there is nothing here for the per-event rows to feed.
            suffix_summary = self.suffix_encoder(
                activities=item.suffix.activities,
                resources=item.suffix.resources,
                timestamps=item.suffix.timestamps,
                pad_mask=padding_mask(
                    lengths=item.suffix_len, seq_len=item.suffix.activities.size(dim=1)
                ),
            )[:, 0]  # [batch_size, d_model]
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

        decoder_output = self.decoder(
            decoder_input=item.decoder_input,
            z=z,
            prefix_encoded=prefix_encoded,
            prefix_pad_mask=prefix_pad_mask,
        )
        return TransformerCVAEOutput(decoder=decoder_output, prior=prior, posterior=posterior)

    @torch.no_grad()
    def generate(self, item: SuffixItem, *, num_samples: int) -> GeneratedSuffix:
        """Generate `num_samples` suffixes for every prefix in `item`.

        The suffix is unknown here, so nothing but the prefix is read: the suffix encoder and
        the posterior stay unrun and every latent comes from `p(z | prefix)`. The samples of
        one prefix differ only in that latent, since the decoder reads its heads greedily,
        which is what makes the spread across them a property of the prior rather than of a
        softmax.

        The prefix is encoded once and its rows repeated, so a sample costs the decoder's
        per-step path and nothing else, and the whole `batch_size * num_samples` is generated as
        one batch rather than as a loop over samples.

        Args:
            item: A batch from `SuffixDataset`, read for its prefix only.
            num_samples: How many suffixes to draw per prefix.
        Returns:
            The generated suffixes, `[batch_size, num_samples, steps]`, with row `(i, j)` the
            j-th sample for the i-th prefix of the batch.
        """
        prefix_pad_mask = padding_mask(
            lengths=item.prefix_len, seq_len=item.prefix.activities.size(dim=1)
        )
        prefix_encoded = self.prefix_encoder(
            activities=item.prefix.activities,
            resources=item.prefix.resources,
            timestamps=item.prefix.timestamps,
            pad_mask=prefix_pad_mask,
        )  # [batch_size, 1 + seq_len, d_model]

        # Every sample of a prefix gets its own row, adjacent to its siblings, so the flat
        # result reshapes straight back into [batch_size, num_samples, ...].
        prefix_encoded = prefix_encoded.repeat_interleave(repeats=num_samples, dim=0)
        prefix_pad_mask = prefix_pad_mask.repeat_interleave(repeats=num_samples, dim=0)

        # The rows of a prefix hold the same distribution, so they differ only in the draw.
        z = self.prior(prefix_encoded[:, 0]).sample()  # [batch_size * num_samples, latent_dim]

        # A suffix holds at most `max_seq_len` events, the padded width the batch comes in at.
        generated = self.decoder.generate(
            z=z,
            prefix_encoded=prefix_encoded,
            prefix_pad_mask=prefix_pad_mask,
            max_steps=item.prefix.activities.size(dim=1),
        )
        batch_size = item.prefix_len.size(dim=0)
        return GeneratedSuffix(
            activities=generated.activities.view(batch_size, num_samples, -1),
            resources=generated.resources.view(batch_size, num_samples, -1),
            timestamps=generated.timestamps.view(batch_size, num_samples, -1),
            lengths=generated.lengths.view(batch_size, num_samples),
        )
