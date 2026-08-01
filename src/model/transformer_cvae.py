from dataclasses import dataclass
import torch
from torch import nn

from src.configs.dataset_info import DatasetInfo
from src.configs.schema import ModelConfig
from src.datasets.dataset import SuffixItem
from src.distributions.gaussian import Gaussian
from src.model.components.decoder import Decoder, DecoderOutput, GeneratedSuffix
from src.model.components.embeddings import EventEmbeddings
from src.model.components.latent import PosteriorNetwork, PriorNetwork
from src.model.components.trace_encoder import TraceEncoder, padding_mask


@dataclass
class TransformerCVAEOutput:
    decoder: DecoderOutput
    prior: Gaussian
    posterior: Gaussian


class TransformerCVAE(nn.Module):
    """
    A conditional VAE that predicts a trace's suffix from its prefix.

    The prefix is the condition, and the decoder cross-attends over it at every position and
    in every layer: over each of its events, and over the CLS row summarizing the whole of it.
    
    Because the decoder can read the prefix directly and the prior is conditioned on it too, the
    latent z is left encoding only what the prefix does not determine.

    Flow, with `prefix` and `suffix` both padded to `max_seq_len`:
        prefix              -> prefix_encoded (a row per event, plus a CLS summary row)
        prefix summary      -> p(z | prefix)               (scored by the KL term only)
        + suffix summary    -> q(z | prefix, suffix)
        z ~ q(z | prefix, suffix)
        z, prefix_encoded, suffix -> activity predictions, and the case's remaining time
    """

    def __init__(self, config: ModelConfig, dataset_info: DatasetInfo):
        super().__init__()
        # The embeddings are shared between the prefix and suffix encoders, and between the decoder
        # Allows to learn a single embedding space for activities and resources, reducing the number of parameters and improving generalization.
        self.embeddings = EventEmbeddings(
            config=config.embeddings, dataset_info=dataset_info, d_model=config.d_model
        )
        # Same architecture, separate weights: one reads prefixes, the other ground-truth suffixes. 
        # The prefix encoder is used at inference, the suffix encoder is not.
        self.prefix_encoder = TraceEncoder(
            config=config.prefix_encoder, embeddings=self.embeddings, d_model=config.d_model
        )
        self.suffix_encoder = TraceEncoder(
            config=config.suffix_encoder, embeddings=self.embeddings, d_model=config.d_model
        )
        # Both prefix and suffix encoders produce a CLS row summarizing the whole of their input (dim=d_model).
        self.prior = PriorNetwork(
            config=config.prior, latent_config=config.latent, prefix_dim=config.d_model
        )
        self.posterior = PosteriorNetwork(
            latent_config=config.latent, prefix_dim=config.d_model, suffix_dim=config.d_model
        )
        # The decoder reads the prefix and the latent z, and predicts the suffix. It is used both at training and inference. 
        self.decoder = Decoder(
            config=config.decoder,
            latent_config=config.latent,
            embeddings=self.embeddings,
            d_model=config.d_model,
            num_activities=dataset_info.num_activities,
            sos_activity_index=dataset_info.sos_activity_index,
            pad_resource_index=dataset_info.pad_resource_index,
            eot_activity_index=dataset_info.eot_activity_index,
        )
        self.pad_activity_index = dataset_info.pad_activity_index
        self.eot_activity_index = dataset_info.eot_activity_index

    @classmethod
    def from_checkpoint(
        cls, checkpoint: dict, dataset_info: DatasetInfo, *, device: str = 'cpu'
    ) -> 'TransformerCVAE':
        """
        Rebuild the model a checkpoint holds, with its weights loaded.
        
        Args:
            checkpoint: A checkpoint read by `load_checkpoint`.
            dataset_info: The dataset the model is to be used on, supplying the vocabulary
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

        model = cls(config=config, dataset_info=dataset_info).to(device=device)
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
        # Mask the padding positions of the prefix, blocking them from being attended over
        prefix_pad_mask = padding_mask(
            lengths=item.prefix_len, seq_len=item.prefix.activities.size(dim=1)
        )  # [batch_size, 1 + seq_len]

        # Encode the prefix, producing a row per event for the decoder to attend over,
        # and a CLS row summarizing the whole of it for the latent networks to read
        prefix_encoded = self.prefix_encoder(
            events=item.prefix, pad_mask=prefix_pad_mask
        )  # [batch_size, 1 + seq_len, d_model]
        prefix_CLS = prefix_encoded[:, 0]  # [batch_size, d_model], CLS row summarizing the whole prefix

        # Encode the suffix, producing a CLS row summarizing the whole of it for the latent networks to read
        suffix_CLS = self.suffix_encoder(
            events=item.suffix,
            pad_mask=padding_mask(
                lengths=item.suffix_len, seq_len=item.suffix.activities.size(dim=1)
            ),
        )[:, 0]  # [batch_size, d_model]

        # p(z | prefix)
        prior = self.prior(prefix_CLS)  # mean, logvar: [batch_size, 2 * latent_dim]
        # q(z | prefix, suffix)
        posterior = self.posterior(prefix_summary=prefix_CLS, suffix_summary=suffix_CLS)
        # Sample a latent z from the posterior
        z = posterior.sample()  # [batch_size, latent_dim]

        # Run the decoder, which reads the prefix and the latent z, and predicts the suffix.
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
        The suffix is unknown here, so only the prefix is encoded and every latent comes from `p(z | prefix)`. 
        
        The samples of one prefix differ only in that latent, since the decoder reads its heads greedily.

        Args:
            item: A batch from `SuffixDataset`, read for its prefix only.
            num_samples: How many suffixes to draw per prefix.
        Returns:
            The generated suffixes, `[batch_size, num_samples, steps]`, with row `(i, j)` the
            j-th sample for the i-th prefix of the batch.
        """
        # Mask the padding positions of the prefix, blocking them from being attended over
        prefix_pad_mask = padding_mask(
            lengths=item.prefix_len, seq_len=item.prefix.activities.size(dim=1)
        )
        # Encode the prefix, producing a row per event for the decoder to attend over
        prefix_encoded = self.prefix_encoder(
            events=item.prefix, pad_mask=prefix_pad_mask
        )  # [batch_size, 1 + seq_len, d_model]

        # Every sample of a prefix gets its own row, adjacent to its siblings, so the flat
        # result reshapes straight back into [batch_size, num_samples, ...].
        prefix_encoded = prefix_encoded.repeat_interleave(repeats=num_samples, dim=0)
        prefix_pad_mask = prefix_pad_mask.repeat_interleave(repeats=num_samples, dim=0)

        # Define the latent for every sample of every prefix. The prior is conditioned on the prefix, so it is a distribution over what the suffix can be given the prefix
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
            lengths=generated.lengths.view(batch_size, num_samples),
            remaining_time=generated.remaining_time.view(batch_size, num_samples),
        )
