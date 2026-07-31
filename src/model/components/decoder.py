from dataclasses import dataclass
import torch
from torch import nn

from src.configs.schema import DecoderConfig, LatentConfig
from src.datasets.dataset import EncodedEvents
from src.model.components.embeddings import EventEmbeddings
from src.model.components.latent import LOGVAR_MIN, LOGVAR_MAX


@dataclass
class DecoderOutput:
    """What the decoder predicts for every suffix position.

    The time delta comes back as a distribution rather than a point: a mean and a
    log-variance, which `loss.masked_gaussian_nll` reads as a Gaussian. That keeps the term a
    log-likelihood in nats, the same units as the activity cross-entropy it is added to, and
    lets the model widen the variance on a gap it cannot call instead of regressing towards
    the middle of a distribution that has an automated follow-up at one end and an overnight
    wait at the other.
    """
    activity_logits: torch.Tensor      # [batch_size, seq_len, num_activities]
    time_delta_mean: torch.Tensor      # [batch_size, seq_len], in [0, 1] like the targets
    time_delta_logvar: torch.Tensor    # [batch_size, seq_len]


@dataclass
class GeneratedSuffix:
    """A batch of freely generated suffixes: what the decoder produced when fed its own
    predictions rather than a ground truth.

    The events are kept as the raw prediction, EOT and everything after it included; `lengths`
    is what says where each suffix actually ended. The leading axes are whatever the caller
    generated over: `[batch_size, steps]` from `Decoder.generate`, and
    `[batch_size, num_samples, steps]` from `TransformerCVAE.generate`.
    """
    activities: torch.Tensor  # [..., steps]
    time_deltas: torch.Tensor  # [..., steps], in [0, 1] like the targets
    lengths: torch.Tensor     # [...], events emitted before EOT, or `steps` if EOT never came


class Decoder(nn.Module):
    """
    Writes the suffix with a transformer decoder: causal self-attention over the suffix so far,
    cross-attention over the encoded prefix.

    The causal mask allows the decoder to self-attend over the suffix so far, but not over anything after it.
    This allows the teacher-forced pass to be a single parallel call rather than a loop.
    
    The prefix reaches every position and every layer through cross-attention, including the CLS
    row that summarizes it.

    z is added to every position of the decoder's input.

    An event it writes is an activity and a time delta. The resource is not among them: which
    clerk picks a case up next is close to unpredictable, and on bpic-2017 a resource head
    scores no better than a first-order Markov chain while taking two thirds of the gradient.
    The prefix encoder still reads resources, where they are free conditioning; only the
    suffix side gives them up.
    """

    def __init__(
        self,
        config: DecoderConfig,
        latent_config: LatentConfig,
        embeddings: EventEmbeddings,
        *,
        d_model: int,
        num_activities: int,
        sos_activity_index: int,
        pad_resource_index: int,
        eot_activity_index: int,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.dropout = nn.Dropout(p=config.dropout)
        
        self.sos_activity_index = sos_activity_index
        self.pad_resource_index = pad_resource_index
        self.eot_activity_index = eot_activity_index

        # Brings the latent up to the width of the residual stream it is added into.
        self.latent_projection = nn.Linear(
            in_features=latent_config.latent_dim, out_features=d_model
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=config.num_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                batch_first=True,
                # Pre-norm, as in the encoders: no warmup schedule exists to make post-norm safe.
                norm_first=True,
            ),
            num_layers=config.num_layers,
            norm=nn.LayerNorm(normalized_shape=d_model),
        )

        # A trunk shared by both heads, so the heads can be smaller and the model can be trained with a single loss.
        self.shared_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
        )
        # One head per field of an event; the timestamp is a Gaussian, so its head is width 2.
        self.activity_head = nn.Linear(
            in_features=config.head_hidden_dim, out_features=num_activities
        )
        self.time_delta_head = nn.Linear(in_features=config.head_hidden_dim, out_features=2)

    def forward(
        self,
        decoder_input: EncodedEvents,
        z: torch.Tensor,
        prefix_encoded: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
    ) -> DecoderOutput:
        """Predict an event for every position of a suffix at once.

        Args:
            decoder_input: The step inputs, `[batch_size, seq_len]` per field. Teacher forcing
                hands the ground-truth suffix shifted one step behind SOS; `generate` hands the
                model's own predictions so far. This pass cannot tell the two apart.
            z: The sampled latent, `[batch_size, latent_dim]`.
            prefix_encoded: The prefix encoder's output, `[batch_size, 1 + prefix_seq_len, d_model]`.
            prefix_pad_mask: True where a prefix position holds padding, matching
                `prefix_encoded`'s middle axis.
        Returns:
            The per-position predictions.
        """
        seq_len = decoder_input.activities.size(dim=1)

        # The decoder is not allowed to read the channel it cannot write. Under teacher forcing
        # the ground-truth resources are sitting in `decoder_input` for the taking, and
        # `generate` has none to feed; blanking them here, in the one call both paths go
        # through, is what keeps the two identical. PAD is the row `EventEmbeddings` holds at a
        # fixed zero vector, so this contributes nothing and collects no gradient.
        events = decoder_input._replace(
            resources=torch.full_like(
                input=decoder_input.resources, fill_value=self.pad_resource_index
            )
        )

        # z is broadcast over positions: the same latent added to every one of them.
        target = self.embeddings(events) + self.latent_projection(z).unsqueeze(
            dim=1
        )  # [batch_size, seq_len, d_model]

        # True above the diagonal, so a position attends over itself and everything before it and
        # nothing after. This is what a suffix being written one event at a time amounts to.
        causal_mask = torch.triu(
            input=torch.ones(
                size=(seq_len, seq_len), dtype=torch.bool, device=target.device
            ),
            diagonal=1,
        )
        # No `tgt_key_padding_mask`: under the causal mask a padded target position is visible
        # only to later positions, which are themselves padded and already dropped from the loss
        # by `ignore_index` and `suffix_len`. Masking them here would instead leave a row with
        # nothing to attend to, whose softmax is a NaN.
        output = self.decoder(
            tgt=self.dropout(target),
            memory=prefix_encoded,
            tgt_mask=causal_mask,
            memory_key_padding_mask=prefix_pad_mask,
        )  # [batch_size, seq_len, d_model]

        # The trunk and the heads are position-wise, so one call serves the whole suffix.
        features = self.shared_layer(output)  # [batch_size, seq_len, head_hidden_dim]
        time_delta = self.time_delta_head(features)  # [batch_size, seq_len, 2]
        return DecoderOutput(
            activity_logits=self.activity_head(features),
            # Targets are min-max normalized into [0, 1], so the mean is squashed to match; the
            # log-variance is a spread rather than a value and is left unsquashed, clamped only
            # against the `exp` that turns an unbounded head into a NaN loss.
            time_delta_mean=time_delta[..., 0].sigmoid(),
            time_delta_logvar=time_delta[..., 1].clamp(min=LOGVAR_MIN, max=LOGVAR_MAX),
        )

    def generate(
        self,
        z: torch.Tensor,
        prefix_encoded: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        max_steps: int,
    ) -> GeneratedSuffix:
        """Run the decoder free, feeding each step the event the previous one predicted.

        Every step is a call to `forward`, over the events written so far, reading the last
        position of what comes back. The masking, the latent conditioning and the heads
        therefore exist in exactly one place, and the path a suffix is generated on is by
        construction the path it was trained on. The heads are read greedily, so the only thing
        that differs between two generations of one prefix is the z each was given, and a spread
        across them is a spread in `p(z | prefix)` rather than in a softmax sample.

        A row that has emitted EOT keeps being stepped until every row has, or `max_steps` is
        reached; `lengths` is what marks its ending, and the events past it are ignored rather
        than suppressed.

        Args:
            z: The sampled latent, `[batch_size, latent_dim]`.
            prefix_encoded: The prefix encoder's output, `[batch_size, 1 + prefix_seq_len, d_model]`.
            prefix_pad_mask: True where a prefix position holds padding, matching
                `prefix_encoded`'s middle axis.
            max_steps: Hard cap on the suffix length, for the prefixes whose generation never
                emits EOT at all.
        Returns:
            The generated suffixes and the length of each.
        """
        batch_size = z.size(dim=0)

        # What the decoder reads. Position 0 is SOS with a timestamp of 0.0, exactly how
        # `SuffixDataset` builds the teacher-forced position 0; position `i` is filled in with
        # the event predicted at step `i - 1`. The resource channel stays PAD throughout, which
        # is what `forward` blanks it to on the teacher-forced path as well.
        input_activities = torch.full(
            size=(batch_size, max_steps),
            fill_value=self.sos_activity_index,
            dtype=torch.long,
            device=z.device,
        )
        input_resources = torch.full(
            size=(batch_size, max_steps),
            fill_value=self.pad_resource_index,
            dtype=torch.long,
            device=z.device,
        )
        input_time_deltas = z.new_zeros(size=(batch_size, max_steps))

        # What it produced, which is the same events shifted one position earlier.
        generated_activities = torch.zeros_like(input=input_activities)
        generated_time_deltas = torch.zeros_like(input=input_time_deltas)

        # A row that never emits EOT ran to the cap, so that is the length it keeps.
        lengths = torch.full(
            size=(batch_size,), fill_value=max_steps, dtype=torch.long, device=z.device
        )
        finished = torch.zeros(size=(batch_size,), dtype=torch.bool, device=z.device)

        steps_taken = max_steps
        for position in range(max_steps):
            # Everything written so far, the whole prefix of the suffix being built.
            output = self.forward(
                decoder_input=EncodedEvents(
                    activities=input_activities[:, : position + 1],
                    resources=input_resources[:, : position + 1],
                    time_deltas=input_time_deltas[:, : position + 1],
                ),
                z=z,
                prefix_encoded=prefix_encoded,
                prefix_pad_mask=prefix_pad_mask,
            )
            # Only the last position is new; the earlier ones just reproduce what is already
            # written, since the causal mask makes them independent of anything after them.
            activities = output.activity_logits[:, -1].argmax(dim=-1)  # [batch_size]
            # The head describes a distribution; its mean is the delta to write down. The
            # log-variance is the model's confidence in it, which nothing downstream reads yet.
            time_deltas = output.time_delta_mean[:, -1]                  # [batch_size]

            generated_activities[:, position] = activities
            generated_time_deltas[:, position] = time_deltas
            # The last step has no next position to feed.
            if position + 1 < max_steps:
                input_activities[:, position + 1] = activities
                input_time_deltas[:, position + 1] = time_deltas

            # A suffix ends at its first EOT, so a later one cannot move the length back.
            just_finished = ~finished & (activities == self.eot_activity_index)
            lengths = lengths.masked_fill(mask=just_finished, value=position)
            finished |= just_finished
            # Reading this stalls the device queue once per step. It pays for itself anyway:
            # suffixes are far shorter than `max_seq_len` on every log here, so most of the loop
            # is skipped outright, and a step of it is a whole decoder pass.
            if bool(finished.all()):
                steps_taken = position + 1
                break

        return GeneratedSuffix(
            activities=generated_activities[:, :steps_taken],  # [batch_size, steps]
            time_deltas=generated_time_deltas[:, :steps_taken],
            lengths=lengths,
        )
