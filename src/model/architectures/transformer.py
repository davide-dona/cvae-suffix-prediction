import torch
import torch.nn.functional as F

from src.configs.schema import TransformerConfig
from src.datasets.codec import DatasetCodec
from src.datasets.dataset import SplitTrace
from src.model.components.decoder import Decoder, GeneratedSuffix
from src.model.components.embeddings import EventEmbeddings
from src.model.components.trace_encoder import TraceEncoder
from src.model.models import ModelOutput, SuffixModel, time_mae
from src.training import LatentMetrics, Loss


class Transformer(SuffixModel):
    """An encoder-decoder transformer that predicts a trace's suffix from its prefix, with no
    latent at all: the same backbone as `TransformerCVAE` with the latent path taken out.

    It is the arm the CVAE is read against, so it is the same everywhere the latent does not
    reach: one embedding space, the same encoder over the prefix, the same decoder
    cross-attending over its events, the same point time heads scored by the same `time_mae`.
    What differs is where the variability lives, and the activity head is the whole of it here:
    with nothing conditioning the decoder, two runs of one prefix could only be the same suffix,
    so the activity is sampled from its logits at every step. That is exactly the per-step noise
    the latent exists to avoid, which is the comparison.

    A cycle time still varies from draw to draw, because it is read off an activity path that was
    drawn. A remaining time does not: it is read at position 0, before any activity has been
    written, so every sample of one prefix opens on the same number. That is what an arm with no
    latent can say about a quantity settled before its first step, and it is a result rather than a
    gap.

    Flow, with `prefix` and `suffix` both padded to `max_seq_len`:
        prefix                -> prefix events (for the decoder) prefix events, suffix -> an
        activity, the cycle time before it and a remaining time, at every
                                 suffix position
    """

    def __init__(self, config: TransformerConfig, codec: DatasetCodec):
        super().__init__(codec=codec)
        self.embeddings = EventEmbeddings(
            config=config.embeddings, codec=codec, d_model=config.d_model
        )
        # Only the prefix is ever read: with no posterior there is nothing a summary of the
        # ground-truth suffix would feed.
        self.encoder = TraceEncoder(
            config=config.encoder, embeddings=self.embeddings, d_model=config.d_model
        )
        self.decoder = Decoder(
            config=config.decoder,
            latent_config=None,
            embeddings=self.embeddings,
            d_model=config.d_model,
            num_activities=codec.activity.num_rows,
            sos_activity_index=codec.activity.sos_index,
            pad_activity_index=codec.activity.pad_index,
            pad_resource_index=codec.resource.pad_index,
            eot_activity_index=codec.activity.eot_index,
            # A sampler: with no z to draw, the activity head is where the variability comes from.
            sampling=config.sampling,
        )

    def forward(self, item: SplitTrace) -> ModelOutput:
        """
        Args:
            item: A batch from `TraceDataset`, read for its prefix and for the suffix the
                decoder is teacher-forced on.
        Returns:
            The decoder's predictions, and no latents: there is no KL term to charge.
        """
        prefix_pad_mask = item.prefix.pad_mask()  # [batch_size, seq_len]
        prefix = self.encoder(events=item.prefix, pad_mask=prefix_pad_mask)

        decoder_output = self.decoder(
            suffix_activities=item.suffix.activities,
            z=None,
            prefix_encoded=prefix.events,
            prefix_pad_mask=prefix_pad_mask,
        )
        return ModelOutput(decoder=decoder_output, latents=None)

    @torch.no_grad()
    def generate(
        self, item: SplitTrace, *, num_samples: int, sample: bool = True
    ) -> GeneratedSuffix:
        """Generate `num_samples` suffixes for every prefix in `item`.

        Args:
            item: A batch from `TraceDataset`, read for its prefix only.
            num_samples: How many suffixes to draw per prefix. Every row decodes independently,
                so `num_samples` rows of one prefix give `num_samples` different suffixes.
            sample: Whether to sample the heads. False reads each at its mode - the likeliest
                activity, the mean of each time - which is the model's single point prediction,
                and makes every row of a prefix identical.
        Returns:
            The generated suffixes, `[batch_size, num_samples, steps]`, with row `(i, j)` the
            j-th sample for the i-th prefix of the batch.
        """
        prefix_pad_mask = item.prefix.pad_mask()  # [batch_size, seq_len]
        prefix = self.encoder(events=item.prefix, pad_mask=prefix_pad_mask)

        # The prefix is encoded once and repeated per sample, so the decoder writes every sample
        # of the batch in one pass. Unlike the CVAE nothing else is repeated: the noise that
        # separates two rows of a prefix is drawn inside the decode loop, step by step.
        prefix_events = prefix.events.repeat_interleave(repeats=num_samples, dim=0)
        prefix_pad_mask = prefix_pad_mask.repeat_interleave(repeats=num_samples, dim=0)

        generated = self.decoder.generate(
            z=None,
            prefix_encoded=prefix_events,
            prefix_pad_mask=prefix_pad_mask,
            max_steps=item.prefix.activities.size(dim=1),
            sample=sample,
        )
        return self._per_sample(generated, batch_size=item.prefix.length.size(dim=0))

    def compute_loss(
        self, output: ModelOutput, batch: SplitTrace, *, step: int
    ) -> tuple[torch.Tensor, Loss, LatentMetrics | None]:
        """Score a forward pass by its reconstruction alone: no latent, no KL term to charge.

        The same three terms the CVAE charges, summed the same way and scored by the same calls,
        so the two arms' losses differ in the KL and nothing else.
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

        metrics = Loss(
            loss=reconstruction_loss.item(),
            reconstruction_loss=reconstruction_loss.item(),
            activity_loss=activity_loss.item(),
            cycle_time_loss=cycle_time_loss.item(),
            remaining_time_loss=remaining_time_loss.item(),
        )
        return reconstruction_loss / batch_size, metrics, None
