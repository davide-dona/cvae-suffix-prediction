from dataclasses import dataclass

import torch
from torch import nn

from src.configs.schema import DecoderConfig, LatentConfig
from src.datasets.dataset import PaddedEvents
from src.model.components.embeddings import EventEmbeddings


@dataclass
class DecoderOutput:
    """What the decoder predicts for every suffix position."""
    activity_logits: torch.Tensor  # [batch_size, seq_len, num_activities]
    resource_logits: torch.Tensor  # [batch_size, seq_len, num_resources]
    timestamps: torch.Tensor       # [batch_size, seq_len], in [0, 1] like the targets


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
    resources: torch.Tensor   # [..., steps]
    timestamps: torch.Tensor  # [..., steps], in [0, 1] like the targets
    lengths: torch.Tensor     # [...], events emitted before EOT, or `steps` if EOT never came


class Decoder(nn.Module):
    """
    Writes the suffix with a transformer decoder: causal self-attention over the suffix so far,
    cross-attention over the encoded prefix.

    The causal mask is the whole of what makes the prediction at a position depend on nothing
    after it, so the teacher-forced pass is a single parallel call rather than a loop. The
    prefix reaches every position and every layer through cross-attention, including the CLS
    row that summarizes it.

    z is added to every position of the decoder's input. It therefore sits in the residual
    stream from the first layer on and cannot be routed around by an attention pattern, which is
    what stands between a conditional VAE and a decoder that quietly learns to ignore its latent.
    """

    def __init__(
        self,
        config: DecoderConfig,
        latent_config: LatentConfig,
        embeddings: EventEmbeddings,
        *,
        d_model: int,
        num_activities: int,
        num_resources: int,
        sos_activity_index: int,
        sos_resource_index: int,
        eot_activity_index: int,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.dropout = nn.Dropout(p=config.dropout)

        # Only `generate` reads these: the token a suffix opens on, and the one it ends on.
        # Teacher forcing is handed both by the dataset instead, in `decoder_input`.
        self.sos_activity_index = sos_activity_index
        self.sos_resource_index = sos_resource_index
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

        # A trunk shared by the three heads below, not a head itself: what an event is depends
        # on its activity, its resource and its timing together, so they are read off one
        # representation rather than three unrelated projections of the same vector.
        self.shared_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
        )
        # One head per field of an event; the timestamp is a scalar, so its head is width 1.
        self.activity_head = nn.Linear(
            in_features=config.head_hidden_dim, out_features=num_activities
        )
        self.resource_head = nn.Linear(
            in_features=config.head_hidden_dim, out_features=num_resources
        )
        self.timestamp_head = nn.Linear(in_features=config.head_hidden_dim, out_features=1)

    def forward(
        self,
        decoder_input: PaddedEvents,
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

        # z is broadcast over positions: the same latent added to every one of them.
        target = self.embeddings(
            activities=decoder_input.activities,
            resources=decoder_input.resources,
            timestamps=decoder_input.timestamps,
        ) + self.latent_projection(z).unsqueeze(dim=1)  # [batch_size, seq_len, d_model]

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
        return DecoderOutput(
            activity_logits=self.activity_head(features),
            resource_logits=self.resource_head(features),
            # Targets are min-max normalized into [0, 1], so the head is squashed to match.
            timestamps=self.timestamp_head(features).sigmoid().squeeze(dim=-1),
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
        # the event predicted at step `i - 1`.
        input_activities = torch.full(
            size=(batch_size, max_steps),
            fill_value=self.sos_activity_index,
            dtype=torch.long,
            device=z.device,
        )
        input_resources = torch.full(
            size=(batch_size, max_steps),
            fill_value=self.sos_resource_index,
            dtype=torch.long,
            device=z.device,
        )
        input_timestamps = z.new_zeros(size=(batch_size, max_steps))

        # What it produced, which is the same events shifted one position earlier.
        generated_activities = torch.zeros_like(input=input_activities)
        generated_resources = torch.zeros_like(input=input_resources)
        generated_timestamps = torch.zeros_like(input=input_timestamps)

        # A row that never emits EOT ran to the cap, so that is the length it keeps.
        lengths = torch.full(
            size=(batch_size,), fill_value=max_steps, dtype=torch.long, device=z.device
        )
        finished = torch.zeros(size=(batch_size,), dtype=torch.bool, device=z.device)

        steps_taken = max_steps
        for position in range(max_steps):
            # Everything written so far, the whole prefix of the suffix being built.
            output = self.forward(
                decoder_input=PaddedEvents(
                    activities=input_activities[:, : position + 1],
                    resources=input_resources[:, : position + 1],
                    timestamps=input_timestamps[:, : position + 1],
                ),
                z=z,
                prefix_encoded=prefix_encoded,
                prefix_pad_mask=prefix_pad_mask,
            )
            # Only the last position is new; the earlier ones just reproduce what is already
            # written, since the causal mask makes them independent of anything after them.
            activities = output.activity_logits[:, -1].argmax(dim=-1)  # [batch_size]
            resources = output.resource_logits[:, -1].argmax(dim=-1)   # [batch_size]
            timestamps = output.timestamps[:, -1]                      # [batch_size]

            generated_activities[:, position] = activities
            generated_resources[:, position] = resources
            generated_timestamps[:, position] = timestamps
            # The last step has no next position to feed.
            if position + 1 < max_steps:
                input_activities[:, position + 1] = activities
                input_resources[:, position + 1] = resources
                input_timestamps[:, position + 1] = timestamps

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
            resources=generated_resources[:, :steps_taken],
            timestamps=generated_timestamps[:, :steps_taken],
            lengths=lengths,
        )
