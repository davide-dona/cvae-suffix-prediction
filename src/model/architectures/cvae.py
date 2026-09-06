import torch
import torch.nn.functional as F

from src.configs.schema import CVAEConfig
from src.datasets.codec import DatasetCodec
from src.datasets.dataset import SplitTrace
from src.distributions import Gaussian
from src.model.components.decoder import Decoder, GeneratedSuffix
from src.model.components.embeddings import EventEmbeddings
from src.model.components.latent import PosteriorNetwork, PriorNetwork
from src.model.components.trace_encoder import TraceEncoder
from src.model.models import Latents, ModelOutput, SuffixModel, time_mae
from src.training import LatentMetrics, Loss, free_bits_kl, gaussian_kl, linear_warmup_weight


class TransformerCVAE(SuffixModel):
    """A conditional VAE that predicts a trace's suffix from its prefix.

    The prefix is the condition: the decoder cross-attends over its encoded events, and its
    CLS summary feeds the latent networks. Because the decoder reads the prefix directly and
    the prior is conditioned on it too, z is left encoding only what the prefix does not
    determine.

    One encoder reads both sequences, so the two summaries the posterior is handed are two
    reads of one representation rather than two coordinate systems the KL has to reconcile.

    Flow, with `prefix` and `suffix` both padded to `max_seq_len`:
        prefix              -> prefix events (for the decoder) + summary (for the latents)
        prefix summary      -> p(z | prefix)               (scored by the KL term only)
        + suffix summary    -> q(z | prefix, suffix)
        z ~ q(z | prefix, suffix)
        z, prefix events, suffix -> an activity, the cycle time before it and a remaining time,
                                   at every suffix position
    """

    def __init__(self, config: CVAEConfig, codec: DatasetCodec):
        super().__init__(codec=codec)
        self.loss_config = config.loss
        # Shared between the encoder and the decoder: a single embedding space for events,
        # with fewer parameters.
        self.embeddings = EventEmbeddings(
            config=config.embeddings, codec=codec, d_model=config.d_model
        )
        # One stack for both sequences: a prefix and a suffix are the same kind of thing, read
        # the same way, so a summary of either lands in one coordinate system. Only the prefix
        # is encoded at inference, the suffix being what the model is asked to write.
        self.encoder = TraceEncoder(
            config=config.encoder, embeddings=self.embeddings, d_model=config.d_model
        )
        self.prior = PriorNetwork(
            config=config.prior, latent_config=config.latent, prefix_dim=config.d_model
        )
        self.posterior = PosteriorNetwork(latent_config=config.latent, summary_dim=config.d_model)
        # Greedy heads: everything the prefix does not determine is z's to carry, so a suffix is
        # one draw from the latent rather than a sequence of independent per-step draws.
        self.decoder = Decoder(
            config=config.decoder,
            latent_config=config.latent,
            embeddings=self.embeddings,
            d_model=config.d_model,
            num_activities=codec.activity.num_rows,
            sos_activity_index=codec.activity.sos_index,
            pad_activity_index=codec.activity.pad_index,
            pad_resource_index=codec.resource.pad_index,
            eot_activity_index=codec.activity.eot_index,
            # No sampler: the activity head is read at its mode, so a suffix is one draw of z.
            sampling=None,
        )

    def forward(self, item: SplitTrace) -> ModelOutput:
        """
        Args:
            item: A batch from `TraceDataset`, read for both its prefix and its suffix.
        Returns:
            The decoder's predictions and the latent distributions the loss compares.
        """
        prefix_pad_mask = item.prefix.pad_mask()  # [batch_size, seq_len]
        prefix = self.encoder(events=item.prefix, pad_mask=prefix_pad_mask)

        suffix_summary = self.encoder(
            events=item.suffix, pad_mask=item.suffix.pad_mask()
        ).summary  # [batch_size, d_model]

        prior = self.prior(prefix.summary)  # p(z | prefix)
        posterior = self.posterior(prefix_summary=prefix.summary, suffix_summary=suffix_summary)
        z = posterior.sample()  # [batch_size, latent_dim]

        # The decoder reads the prefix events only; the summary belongs to the latent path.
        decoder_output = self.decoder(
            suffix_activities=item.suffix.activities,
            z=z,
            prefix_encoded=prefix.events,
            prefix_pad_mask=prefix_pad_mask,
        )
        return ModelOutput(
            decoder=decoder_output, latents=Latents(prior=prior, posterior=posterior)
        )

    @torch.no_grad()
    def generate(
        self, item: SplitTrace, *, num_samples: int, sample: bool = True
    ) -> GeneratedSuffix:
        """Generate `num_samples` suffixes for every prefix in `item`.

        The suffix is unknown here, so only the prefix is encoded and every latent comes from
        `p(z | prefix)`. The samples of one prefix differ only in that latent, since the
        decoder reads its heads greedily.

        Args:
            item: A batch from `TraceDataset`, read for its prefix only.
            num_samples: How many suffixes to draw per prefix.
            sample: Whether to draw z from `p(z | prefix)`. False takes its mean instead,
                making the generation the model's single point prediction.
        Returns:
            The generated suffixes, `[batch_size, num_samples, steps]`, with row `(i, j)` the
            j-th sample for the i-th prefix of the batch.
        """
        prefix_pad_mask = item.prefix.pad_mask()  # [batch_size, seq_len]
        prefix = self.encoder(events=item.prefix, pad_mask=prefix_pad_mask)

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
        repeated = Gaussian(
            mean=prior.mean.repeat_interleave(repeats=num_samples, dim=0),
            logvar=prior.logvar.repeat_interleave(repeats=num_samples, dim=0),
        )
        z = repeated.sample() if sample else repeated.mean  # [rows, latent_dim]

        # A suffix holds at most `max_seq_len` events, the padded width the batch comes in at.
        # `sample=False` throughout: this decoder's heads are read at their mode whatever is
        # asked of them, the draw having already happened in z.
        generated = self.decoder.generate(
            z=z,
            prefix_encoded=prefix_events,
            prefix_pad_mask=prefix_pad_mask,
            max_steps=item.prefix.activities.size(dim=1),
            sample=False,
        )
        return self._per_sample(generated, batch_size=item.prefix.length.size(dim=0))

    def compute_loss(
        self, output: ModelOutput, batch: SplitTrace, *, step: int
    ) -> tuple[torch.Tensor, Loss, LatentMetrics | None]:
        """Score a forward pass by its ELBO: reconstruction plus the annealed, floored KL.

        The two time heads are point regressors scored by `time_mae`, which is the simple regressor
        this architecture's decoder already is. Everything the prefix leaves open is z's to carry,
        the spread of a cycle time included, so nothing here is handed a scale of its own to widen
        instead.
        """
        batch_size = batch.suffix.activities.size(0)

        activity_loss = F.cross_entropy(
            output.decoder.activity_logits.transpose(1, 2),
            batch.suffix.activities,
            ignore_index=self.pad_activity_index,
            reduction='sum',
        )

        cycle_time_loss = time_mae(output.decoder.cycle_times, batch.cycle_times, batch)
        remaining_time_loss = time_mae(output.decoder.remaining_times, batch.remaining_times, batch)

        reconstruction_loss = activity_loss + cycle_time_loss + remaining_time_loss

        kl_per_dim = gaussian_kl(
            posterior=output.latents.posterior, prior=output.latents.prior
        )  # [batch_size, latent_dim]
        floored_kl_loss = free_bits_kl(kl_per_dim, free_bits=self.loss_config.free_bits)
        kl_weight = linear_warmup_weight(
            step,
            ramp_steps=self.loss_config.kl_annealing_ramp_steps,
            start=self.loss_config.kl_annealing_start_weight,
            stop=self.loss_config.kl_annealing_full_weight,
        )
        total_loss = reconstruction_loss + kl_weight * floored_kl_loss

        metrics = Loss(
            loss=total_loss.item(),
            reconstruction_loss=reconstruction_loss.item(),
            floored_kl_loss=floored_kl_loss.item(),
            activity_loss=activity_loss.item(),
            cycle_time_loss=cycle_time_loss.item(),
            remaining_time_loss=remaining_time_loss.item(),
        )
        latent = LatentMetrics.of(
            kl_per_dim, free_bits=self.loss_config.free_bits, kl_weight=kl_weight
        )
        return total_loss / batch_size, metrics, latent
