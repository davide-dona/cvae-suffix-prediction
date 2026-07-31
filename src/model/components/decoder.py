from dataclasses import dataclass
import torch
from torch import nn

from src.configs.schema import DecoderConfig, LatentConfig
from src.datasets.dataset import EncodedEvents
from src.model.components.attention import MultiHeadAttention, ProjectedKeysValues
from src.model.components.embeddings import EventEmbeddings
from src.model.components.latent import LOGVAR_MIN, LOGVAR_MAX


@dataclass(frozen=True)
class LayerCache:
    """What one decoder layer does not have to project again to write the next event.

    `memory` is the encoded prefix, which does not change over a whole suffix. `written` is the
    suffix positions already emitted, which grows by one per event and never changes behind the
    front. Between them they are everything a step would otherwise recompute from scratch, which
    is what turns generating a suffix of n events from n passes over n positions into n passes
    over one.
    """
    memory: ProjectedKeysValues
    written: ProjectedKeysValues


class DecoderLayer(nn.Module):
    """One pre-norm decoder layer: causal self-attention, cross-attention over the prefix, and a
    feed-forward block, each entered through a norm and left through a residual add.

    Pre-norm, as in the encoders: no warmup schedule exists here to make post-norm safe.

    The same layer serves both passes. Handed no cache it reads a whole suffix at once under a
    causal mask, which is the teacher-forced pass. Handed one it reads the single event that
    follows what the cache holds. Nothing about the arithmetic differs between the two, which is
    what keeps the path a suffix is generated on the path it was trained on.

    A cached call reads exactly one new position: it drops the causal mask, which is only
    correct when every key in the cache precedes the query.
    """

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
        target: torch.Tensor,
        *,
        memory: torch.Tensor,
        memory_pad_mask: torch.Tensor,
        cache: LayerCache | None,
    ) -> tuple[torch.Tensor, LayerCache]:
        """
        Args:
            target: The positions to read, `[batch_size, seq_len, d_model]`. The whole suffix
                without a cache, the one event that follows it with one.
            memory: The prefix encoder's output, `[batch_size, 1 + prefix_seq_len, d_model]`.
            memory_pad_mask: True where a prefix position holds padding.
            cache: What a previous call projected, or None to project everything.
        Returns:
            The layer's output for the positions read, and the cache to hand the next call.
        """
        normalized = self.self_attention_norm(target)
        written = self.self_attention.project(normalized)
        if cache is not None:
            written = cache.written.extend(written)
        target = target + self.dropout(
            self.self_attention(query=normalized, keys_values=written, causal=cache is None)
        )

        # The prefix is projected on the first call of a suffix and read back on every one after.
        memory_keys_values = (
            cache.memory if cache is not None else self.cross_attention.project(memory)
        )
        target = target + self.dropout(
            self.cross_attention(
                query=self.cross_attention_norm(target),
                keys_values=memory_keys_values,
                key_padding_mask=memory_pad_mask,
            )
        )

        target = target + self.dropout(self.feedforward(self.feedforward_norm(target)))
        return target, LayerCache(memory=memory_keys_values, written=written)


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
        self.layers = nn.ModuleList(
            DecoderLayer(config, d_model=d_model) for _ in range(config.num_layers)
        )
        # Pre-norm leaves the last layer's residual stream unnormalized, so the stack closes
        # with a norm of its own, as `nn.TransformerDecoder`'s `norm` argument did.
        self.norm = nn.LayerNorm(normalized_shape=d_model)

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
        hidden, _ = self._run_layers(
            events=decoder_input,
            z=z,
            memory=prefix_encoded,
            memory_pad_mask=prefix_pad_mask,
            start_position=0,
            caches=None,
        )  # [batch_size, seq_len, d_model]
        return self._predict(hidden)

    def _run_layers(
        self,
        *,
        events: EncodedEvents,
        z: torch.Tensor,
        memory: torch.Tensor,
        memory_pad_mask: torch.Tensor,
        start_position: int,
        caches: list[LayerCache] | None,
    ) -> tuple[torch.Tensor, list[LayerCache]]:
        """Embed a run of decoder inputs and push it through the stack.

        The one place both passes go through, so what generation runs is by construction what
        training ran. `caches` is the only thing that differs between them: without one this
        reads a whole suffix under a causal mask, with one it reads the single event that
        follows what the caches already hold.

        Args:
            events: The decoder inputs to read, `[batch_size, seq_len]` per field.
            z: The sampled latent, `[batch_size, latent_dim]`.
            memory: The prefix encoder's output, `[batch_size, 1 + prefix_seq_len, d_model]`.
            memory_pad_mask: True where a prefix position holds padding.
            start_position: Where in the suffix `events` starts, for the positional encoding.
            caches: One per layer, from a previous call, or None to read from the beginning.
        Returns:
            The stack's output for the positions read, and the caches carrying them.
        """
        # The decoder is not allowed to read the channel it cannot write. Under teacher forcing
        # the ground-truth resources are sitting in `events` for the taking, and `generate` has
        # none to feed; blanking them here, in the one call both paths go through, is what keeps
        # the two identical. PAD is the row `EventEmbeddings` holds at a fixed zero vector, so
        # this contributes nothing and collects no gradient.
        blanked = events._replace(
            resources=torch.full_like(
                input=events.resources, fill_value=self.pad_resource_index
            )
        )

        # z is broadcast over positions: the same latent added to every one of them.
        target = self.dropout(
            self.embeddings(blanked, start_position=start_position)
            + self.latent_projection(z).unsqueeze(dim=1)
        )  # [batch_size, seq_len, d_model]

        # No target padding mask: under the causal mask a padded position is visible only to
        # later positions, which are themselves padded and already dropped from the loss by
        # `ignore_index` and `suffix_len`. Masking them here would instead leave a row with
        # nothing to attend to, whose softmax is a NaN.
        layer_caches = caches if caches is not None else [None] * len(self.layers)
        updated: list[LayerCache] = []
        for layer, cache in zip(self.layers, layer_caches):
            target, layer_cache = layer(
                target, memory=memory, memory_pad_mask=memory_pad_mask, cache=cache
            )
            updated.append(layer_cache)

        return self.norm(target), updated

    def _predict(self, hidden: torch.Tensor) -> DecoderOutput:
        """Read the stack's output as an event per position.

        Args:
            hidden: `[batch_size, seq_len, d_model]`.
        Returns:
            The per-position predictions.
        """
        # The trunk and the heads are position-wise, so one call serves however many positions
        # it is handed.
        features = self.shared_layer(hidden)  # [batch_size, seq_len, head_hidden_dim]
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

        Every step is a call to `_run_layers`, the same one the teacher-forced pass makes, so
        the masking, the latent conditioning and the heads exist in exactly one place and the
        path a suffix is generated on is by construction the path it was trained on. What a step
        hands it is the one event just written, plus the caches holding everything before it:
        the encoded prefix, projected once for the whole suffix, and the suffix positions
        already emitted, each projected once rather than once for every event that follows it.
        Writing n events is then n passes over one position instead of n passes over n.

        The heads are read greedily, so the only thing that differs between two generations of
        one prefix is the z each was given, and a spread across them is a spread in
        `p(z | prefix)` rather than in a softmax sample.

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
        caches: list[LayerCache] | None = None
        for position in range(max_steps):
            # Only this one position is new. Everything before it is in `caches`, and under the
            # causal mask it could not have changed anyway.
            hidden, caches = self._run_layers(
                events=EncodedEvents(
                    activities=input_activities[:, position : position + 1],
                    resources=input_resources[:, position : position + 1],
                    time_deltas=input_time_deltas[:, position : position + 1],
                ),
                z=z,
                memory=prefix_encoded,
                memory_pad_mask=prefix_pad_mask,
                start_position=position,
                caches=caches,
            )
            output = self._predict(hidden)

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
            # is skipped outright.
            if bool(finished.all()):
                steps_taken = position + 1
                break

        return GeneratedSuffix(
            activities=generated_activities[:, :steps_taken],  # [batch_size, steps]
            time_deltas=generated_time_deltas[:, :steps_taken],
            lengths=lengths,
        )
