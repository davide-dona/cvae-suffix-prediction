from dataclasses import dataclass
import torch
from torch import nn

from src.configs.schema import DecoderConfig, LatentConfig
from src.datasets.dataset import EncodedEvents
from src.distributions.gaussian import Gaussian
from src.model.components.attention import MultiHeadAttention, ProjectedKeysValues
from src.model.components.embeddings import EventEmbeddings

REMAINING_TIME_LOGVAR_MIN = -6.0


@dataclass(frozen=True)
class LayerCache:
    """A KV cache for one decoder layer: the projected prefix, and the suffix positions already read."""
    prefix_kv: ProjectedKeysValues
    suffix_kv: ProjectedKeysValues


@dataclass(frozen=True)
class DecoderOutput:
    """What the decoder predicts: an activity per suffix position, and one remaining-time
    distribution for the whole suffix, read off position 0 (the state after SOS).

    The remaining time is a Gaussian rather than a point, so its loss is a log-likelihood in
    nats, the same units as the activity cross-entropy it is added to.
    """
    activity_logits: torch.Tensor   # [batch_size, seq_len, num_activities]
    remaining_time_distr: Gaussian  # [batch_size], mean and log-variance


@dataclass(frozen=True)
class GeneratedSuffix:
    """A batch of freely generated suffixes, kept as the raw predictions: EOT and everything
    after it included, with `lengths` saying where each suffix actually ended.

    The leading axes are whatever the caller generated over: `[batch_size, ...]` from
    `Decoder.generate`, `[batch_size, num_samples, ...]` from `TransformerCVAE.generate`.
    """
    activities: torch.Tensor      # [..., steps]
    lengths: torch.Tensor         # [...], events emitted before EOT, or `steps` if EOT never came
    remaining_time: torch.Tensor  # [...], in [0, 1] like the targets


class DecoderLayer(nn.Module):
    """One layer of the decoder stack, with self-attention over the suffix and cross-attention over the prefix."""

    def __init__(self, config: DecoderConfig, *, d_model: int):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            d_model=d_model, num_heads=config.num_heads, dropout=config.dropout
        )
        self.cross_attention = MultiHeadAttention(
            d_model=d_model, num_heads=config.num_heads, dropout=config.dropout
        )
        self.feedforward = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=config.feedforward_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(in_features=config.feedforward_dim, out_features=d_model),
        )
        self.self_attention_norm = nn.LayerNorm(normalized_shape=d_model)
        self.cross_attention_norm = nn.LayerNorm(normalized_shape=d_model)
        self.feedforward_norm = nn.LayerNorm(normalized_shape=d_model)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        prefix_encoded: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        cache: LayerCache | None,
    ) -> tuple[torch.Tensor, LayerCache]:
        """Run one layer over the suffix positions in `hidden`.

        Args:
            hidden: The positions to read, `[batch_size, seq_len, d_model]`: the whole suffix
                without a cache, the one event that follows it with one.
            prefix_encoded: The encoded prefix events, `[batch_size, prefix_seq_len, d_model]`.
            prefix_pad_mask: True where a prefix position holds padding.
            cache: What a previous call projected, or None to project everything.
        Returns:
            The layer's output for the positions read, and the cache to hand the next call.
        """
        hidden_norm = self.self_attention_norm(hidden)
        suffix_kv = self.self_attention.project(hidden_norm)
        # With a cache, the new projections extend the suffix positions already read.
        if cache is not None:
            suffix_kv = cache.suffix_kv.extend(suffix_kv)
        hidden = hidden + self.dropout(
            self.self_attention(query=hidden_norm, keys_values=suffix_kv, causal=cache is None)
        )

        # The prefix is projected on the first call of a suffix and read back on every one after.
        prefix_kv = (
            cache.prefix_kv if cache is not None else self.cross_attention.project(prefix_encoded)
        )
        hidden = hidden + self.dropout(
            self.cross_attention(
                query=self.cross_attention_norm(hidden),
                keys_values=prefix_kv,
                key_padding_mask=prefix_pad_mask,
            )
        )

        hidden = hidden + self.dropout(self.feedforward(self.feedforward_norm(hidden)))
        return hidden, LayerCache(prefix_kv=prefix_kv, suffix_kv=suffix_kv)


class Decoder(nn.Module):
    """Transformer decoder that predicts a suffix of events, given the prefix and a latent z.
    
    Applies self-attention over the suffix positions read so far, cross-attention over 
    the encoded prefix. The latent z is projected and added to every suffix position"""

    def __init__(
        self,
        config: DecoderConfig,
        latent_config: LatentConfig,
        embeddings: EventEmbeddings,
        *,
        d_model: int,
        num_activities: int,
        sos_activity_index: int,
        pad_activity_index: int,
        pad_resource_index: int,
        eot_activity_index: int,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.dropout = nn.Dropout(p=config.dropout)
        self.activity_dropout = config.activity_dropout

        self.sos_activity_index = sos_activity_index
        self.pad_activity_index = pad_activity_index
        self.pad_resource_index = pad_resource_index
        self.eot_activity_index = eot_activity_index

        # Brings the latent up to the width of the residual stream it is added into.
        self.latent_projection = nn.Linear(
            in_features=latent_config.latent_dim, out_features=d_model
        )
        self.layers = nn.ModuleList(
            DecoderLayer(config, d_model=d_model) for _ in range(config.num_layers)
        )
        # Pre-norm leaves the last layer's residual stream unnormalized, so the stack closes
        # with a norm of its own.
        self.norm = nn.LayerNorm(normalized_shape=d_model)

        # A trunk shared by both heads, so the heads can be smaller.
        self.shared_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
        )
        # The remaining time is a Gaussian, so its head is width 2.
        self.activity_head = nn.Linear(
            in_features=config.head_hidden_dim, out_features=num_activities
        )
        self.remaining_time_head = nn.Linear(in_features=config.head_hidden_dim, out_features=2)

    def forward(
        self,
        decoder_input: torch.Tensor,
        z: torch.Tensor,
        prefix_encoded: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
    ) -> DecoderOutput:
        """Predict an event for every position of a suffix at once.

        Args:
            decoder_input: The ground-truth suffix activities shifted one step behind SOS,
                `[batch_size, seq_len]`.
            z: The sampled latent, `[batch_size, latent_dim]`.
            prefix_encoded: The encoded prefix events, `[batch_size, prefix_seq_len, d_model]`.
            prefix_pad_mask: True where a prefix position holds padding.
        Returns:
            The per-position predictions.
        """
        if self.training and self.activity_dropout > 0.0:
            decoder_input = self._drop_activities(decoder_input)
        hidden, _ = self._run_layers(
            activities=decoder_input,
            z=z,
            prefix_encoded=prefix_encoded,
            prefix_pad_mask=prefix_pad_mask,
            start_position=0,
            caches=None,
        )  # [batch_size, seq_len, d_model]
        features = self.shared_layer(hidden)  # [batch_size, seq_len, head_hidden_dim]
        return DecoderOutput(
            activity_logits=self.activity_head(features),
            remaining_time_distr=self._remaining_time_distr(features[:, 0]),
        )

    def _drop_activities(self, activities: torch.Tensor) -> torch.Tensor:
        """Blank a random `activity_dropout` fraction of the teacher-forced activities to PAD.

        A decoder that cannot count on the previous ground-truth token has to look to z for
        what comes next, which keeps information flowing through the latent.
        """
        dropped = (
            torch.rand_like(activities, dtype=torch.float32) < self.activity_dropout
        ) & (activities != self.sos_activity_index)  # [batch_size, seq_len]
        return activities.masked_fill(mask=dropped, value=self.pad_activity_index)

    def _run_layers(
        self,
        *,
        activities: torch.Tensor,
        z: torch.Tensor,
        prefix_encoded: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        start_position: int,
        caches: list[LayerCache] | None,
    ) -> tuple[torch.Tensor, list[LayerCache]]:
        """Embed a run of decoder inputs and push it through the stack.

        The one pass both teacher forcing and `generate` go through; `caches` is all that
        differs between them. Without one this reads a whole suffix under a causal mask, with
        one it reads the single event that follows what the caches already hold.

        Args:
            activities: The decoder input activities to read, `[batch_size, seq_len]`.
            z: The sampled latent, `[batch_size, latent_dim]`.
            prefix_encoded: The encoded prefix events, `[batch_size, prefix_seq_len, d_model]`.
            prefix_pad_mask: True where a prefix position holds padding.
            start_position: Where in the suffix `activities` starts, for the positional encoding.
            caches: One per layer, from a previous call, or None to read from the beginning.
        Returns:
            The stack's output for the positions read, and the caches carrying them.
        """
        # z is broadcast over positions: the same latent added to every one of them.
        hidden = self.dropout(
            self.embeddings(self._blank_events(activities), start_position=start_position)
            + self.latent_projection(z).unsqueeze(dim=1)
        )  # [batch_size, seq_len, d_model]

        # No suffix padding mask: under the causal mask a padded position is visible only to
        # later positions, themselves padded and dropped from the loss. Masking it here would
        # leave a row with nothing to attend to, whose softmax is a NaN.
        layer_caches = caches if caches is not None else [None] * len(self.layers)
        new_caches: list[LayerCache] = []
        for layer, cache in zip(self.layers, layer_caches):
            hidden, layer_cache = layer(
                hidden, prefix_encoded=prefix_encoded, prefix_pad_mask=prefix_pad_mask, cache=cache
            )
            new_caches.append(layer_cache)

        return self.norm(hidden), new_caches

    def _remaining_time_distr(self, feature: torch.Tensor) -> Gaussian:
        """Read the suffix's remaining time off position 0, the state after SOS.

        Args:
            feature: The shared trunk's output at position 0, `[batch_size, head_hidden_dim]`.
        Returns:
            The remaining-time distribution, `[batch_size]` per field.
        """
        parameters = self.remaining_time_head(feature)  # [batch_size, 2]
        # Targets are min-max normalized into [0, 1], so the mean is squashed to match, and
        # the log-variance floor is tightened to match the target's narrow scale.
        return Gaussian.create(
            mean=parameters[..., 0].sigmoid(),
            logvar=parameters[..., 1],
            logvar_min=REMAINING_TIME_LOGVAR_MIN,
        )

    def _blank_events(self, activities: torch.Tensor) -> EncodedEvents:
        """Wrap decoder input activities as the events `EventEmbeddings` reads.

        The decoder has no head to write a resource, a time delta or a feature, so it may not
        read ground truth for them either: teacher forcing would otherwise hand it values
        `generate` has none of to feed, and the two would read different things. Only the
        activities carry real content; every other channel is blanked to the same PAD row or
        0.0 scalar `generate` starts from.

        Args:
            activities: Vocabulary indices to read as the activity channel, `[batch_size, seq_len]`.
        Returns:
            The events, ready for `EventEmbeddings`.
        """
        batch_size, seq_len = activities.shape
        device = activities.device
        return EncodedEvents(
            activities=activities,
            resources=torch.full(
                size=(batch_size, seq_len),
                fill_value=self.pad_resource_index,
                dtype=torch.long,
                device=device,
            ),
            time_deltas=torch.zeros(size=(batch_size, seq_len), device=device),
            feature_categories=torch.zeros(
                size=(batch_size, seq_len, self.embeddings.num_categorical),
                dtype=torch.long,
                device=device,
            ),
            feature_values=torch.zeros(
                size=(batch_size, seq_len, self.embeddings.num_numeric), device=device
            ),
            feature_present=torch.zeros(
                size=(batch_size, seq_len, self.embeddings.num_numeric), device=device
            ),
        )

    def generate(
        self,
        z: torch.Tensor,
        prefix_encoded: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        max_steps: int,
    ) -> GeneratedSuffix:
        """Run the decoder free, feeding each step the event the previous one predicted.

        Every step is one cached call to `_run_layers`, the same pass teacher forcing runs, so
        writing n events costs n passes over one position. The activity head is read greedily
        and the remaining time is its head's mean: two generations of one prefix differ only
        in the z each was given.

        Args:
            z: The sampled latent, `[batch_size, latent_dim]`.
            prefix_encoded: The encoded prefix events, `[batch_size, prefix_seq_len, d_model]`.
            prefix_pad_mask: True where a prefix position holds padding.
            max_steps: Hard cap on the suffix length, for generations that never emit EOT.
        Returns:
            The generated suffixes, the length of each, and the remaining time of each.
        """
        batch_size = z.size(dim=0)

        # What the decoder reads at each step: SOS first, exactly how `SuffixDataset` builds
        # the teacher-forced position 0, then the activity the previous step predicted.
        next_input = torch.full(
            size=(batch_size, 1),
            fill_value=self.sos_activity_index,
            dtype=torch.long,
            device=z.device,
        )

        generated_activities = torch.zeros(
            size=(batch_size, max_steps), dtype=torch.long, device=z.device
        )
        # A row that never emits EOT ran to the cap, so that is the length it keeps.
        lengths = torch.full(
            size=(batch_size,), fill_value=max_steps, dtype=torch.long, device=z.device
        )
        finished = torch.zeros(size=(batch_size,), dtype=torch.bool, device=z.device)

        steps_taken = max_steps
        caches: list[LayerCache] | None = None
        for position in range(max_steps):
            # Only this one position is new; everything before it is in `caches`.
            hidden, caches = self._run_layers(
                activities=next_input,
                z=z,
                prefix_encoded=prefix_encoded,
                prefix_pad_mask=prefix_pad_mask,
                start_position=position,
                caches=caches,
            )
            features = self.shared_layer(hidden[:, 0])  # [batch_size, head_hidden_dim]
            activities = self.activity_head(features).argmax(dim=-1)  # [batch_size]
            # Only position 0, the state after SOS, answers for the whole suffix.
            if position == 0:
                remaining_time = self._remaining_time_distr(features).mean  # [batch_size]

            generated_activities[:, position] = activities
            next_input = activities.unsqueeze(dim=1)  # [batch_size, 1]

            # A suffix ends at its first EOT, so a later one cannot move the length back.
            just_finished = ~finished & (activities == self.eot_activity_index)
            lengths = lengths.masked_fill(mask=just_finished, value=position)
            finished |= just_finished
            # Reading this stalls the device queue once per step, but suffixes are far shorter
            # than `max_steps` on every log here, so most of the loop is skipped outright.
            if bool(finished.all()):
                steps_taken = position + 1
                break

        return GeneratedSuffix(
            activities=generated_activities[:, :steps_taken],  # [batch_size, steps]
            lengths=lengths,
            remaining_time=remaining_time,
        )
