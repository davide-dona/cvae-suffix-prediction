from dataclasses import dataclass
import torch
from torch import nn

from src.datasets.description import DatasetDescription
from src.configs.schema import ModelConfig
from src.datasets.dataset import SuffixItem
from src.distributions.gaussian import Gaussian
from src.model.components.decoder import Decoder, DecoderOutput, GeneratedSuffix
from src.model.components.embeddings import EventEmbeddings
from src.model.components.latent import PosteriorNetwork, PriorNetwork
from src.model.components.trace_encoder import TraceEncoder, padding_mask


@dataclass(frozen=True)
class TransformerCVAEOutput:
    decoder: DecoderOutput
    prior: Gaussian
    posterior: Gaussian


class TransformerCVAE(nn.Module):
    """A conditional VAE that predicts a trace's suffix from its prefix.

    The prefix is the condition: the decoder cross-attends over its encoded events, and its
    CLS summary feeds the latent networks. Because the decoder reads the prefix directly and
    the prior is conditioned on it too, z is left encoding only what the prefix does not
    determine.

    Flow, with `prefix` and `suffix` both padded to `max_seq_len`:
        prefix              -> prefix events (for the decoder) + summary (for the latents)
        prefix summary      -> p(z | prefix)               (scored by the KL term only)
        + suffix summary    -> q(z | prefix, suffix)
        z ~ q(z | prefix, suffix)
        z, prefix events, suffix -> activity predictions, and the case's remaining time
    """

    def __init__(self, config: ModelConfig, description: DatasetDescription):
        super().__init__()
        # Shared between both encoders and the decoder: a single embedding space for events,
        # with fewer parameters.
        self.embeddings = EventEmbeddings(
            config=config.embeddings, description=description, d_model=config.d_model
        )
        # Same architecture, separate weights: one reads prefixes, the other ground-truth
        # suffixes. Only the prefix encoder runs at inference.
        self.prefix_encoder = TraceEncoder(
            config=config.prefix_encoder, embeddings=self.embeddings, d_model=config.d_model
        )
        self.suffix_encoder = TraceEncoder(
            config=config.suffix_encoder, embeddings=self.embeddings, d_model=config.d_model
        )
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
            num_activities=description.activity.num_rows,
            sos_activity_index=description.activity.sos_index,
            pad_activity_index=description.activity.pad_index,
            pad_resource_index=description.resource.pad_index,
            eot_activity_index=description.activity.eot_index,
        )
        self.pad_activity_index = description.activity.pad_index
        self.eot_activity_index = description.activity.eot_index

    @classmethod
    def from_checkpoint(
        cls, checkpoint: dict, description: DatasetDescription, *, device: str = 'cpu'
    ) -> 'TransformerCVAE':
        """Rebuild the model a checkpoint holds, with its weights loaded.

        Args:
            checkpoint: A checkpoint read by `load_checkpoint`.
            description: The dataset the model is to be used on, supplying the vocabulary
                sizes and sequence length it was built against.
            device: Where to place the model.
        Returns:
            The model, in evaluation mode.
        Raises:
            ValueError: If the checkpoint does not carry a config and weights.
        """
        missing = {'model_config', 'model_state_dict'} - checkpoint.keys()
        if missing:
            raise ValueError(f'checkpoint is missing {sorted(missing)}. Train the model again.')

        config = ModelConfig.model_validate(checkpoint['model_config'])

        model = cls(config=config, description=description).to(device=device)
        model.load_state_dict(state_dict=checkpoint['model_state_dict'])
        model.eval()
        return model

    def forward(self, item: SuffixItem) -> TransformerCVAEOutput:
        """
        Args:
            item: A batch from `SuffixDataset`, read for both its prefix and its suffix.
        Returns:
            The decoder's predictions and the latent distributions the loss compares.
        """
        prefix_pad_mask = padding_mask(
            lengths=item.prefix_len, seq_len=item.prefix.activities.size(dim=1)
        )  # [batch_size, seq_len]
        prefix = self.prefix_encoder(events=item.prefix, pad_mask=prefix_pad_mask)
        
        suffix_summary = self.suffix_encoder(
            events=item.suffix,
            pad_mask=padding_mask(
                lengths=item.suffix_len, seq_len=item.suffix.activities.size(dim=1)
            ),
        ).summary  # [batch_size, d_model]

        prior = self.prior(prefix.summary)  # p(z | prefix)
        posterior = self.posterior(prefix_summary=prefix.summary, suffix_summary=suffix_summary)
        z = posterior.sample()  # [batch_size, latent_dim]

        # The decoder reads the prefix events only; the summary belongs to the latent path.
        decoder_output = self.decoder(
            decoder_input=item.decoder_input,
            z=z,
            prefix_encoded=prefix.events,
            prefix_pad_mask=prefix_pad_mask,
        )
        return TransformerCVAEOutput(decoder=decoder_output, prior=prior, posterior=posterior)

    @torch.no_grad()
    def generate(self, item: SuffixItem, *, num_samples: int) -> GeneratedSuffix:
        """Generate `num_samples` suffixes for every prefix in `item`.

        The suffix is unknown here, so only the prefix is encoded and every latent comes from
        `p(z | prefix)`. The samples of one prefix differ only in that latent, since the
        decoder reads its heads greedily.

        Args:
            item: A batch from `SuffixDataset`, read for its prefix only.
            num_samples: How many suffixes to draw per prefix.
        Returns:
            The generated suffixes, `[batch_size, num_samples, steps]`, with row `(i, j)` the
            j-th sample for the i-th prefix of the batch.
        """
        prefix_pad_mask = padding_mask(
            lengths=item.prefix_len, seq_len=item.prefix.activities.size(dim=1)
        )  # [batch_size, seq_len]
        prefix = self.prefix_encoder(events=item.prefix, pad_mask=prefix_pad_mask)

        # Computed once per prefix: every sample of one prefix is drawn from the same
        # p(z | prefix), so running the prior once and repeating its parameters skips
        # num_samples - 1 redundant forward passes over identical rows.
        prior = self.prior(prefix.summary)

        # Repeat the prefix events and pad mask for every sample, so the decoder sees a batch of
        # size `batch_size * num_samples` and can generate all samples in one forward pass
        prefix_events = prefix.events.repeat_interleave(repeats=num_samples, dim=0)
        prefix_pad_mask = prefix_pad_mask.repeat_interleave(repeats=num_samples, dim=0)

        # One latent per sample: independent noise per draw, `Gaussian.sample()`'s
        # reparameterization trick, on top of the one prior distribution repeated per prefix.
        z = Gaussian(
            mean=prior.mean.repeat_interleave(repeats=num_samples, dim=0),
            logvar=prior.logvar.repeat_interleave(repeats=num_samples, dim=0),
        ).sample()  # [batch_size * num_samples, latent_dim]

        # A suffix holds at most `max_seq_len` events, the padded width the batch comes in at.
        generated = self.decoder.generate(
            z=z,
            prefix_encoded=prefix_events,
            prefix_pad_mask=prefix_pad_mask,
            max_steps=item.prefix.activities.size(dim=1),
        )
        batch_size = item.prefix_len.size(dim=0)
        return GeneratedSuffix(
            activities=generated.activities.view(batch_size, num_samples, -1),
            lengths=generated.lengths.view(batch_size, num_samples),
            remaining_time=generated.remaining_time.view(batch_size, num_samples),
        )
