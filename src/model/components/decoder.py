from dataclasses import dataclass

import torch
from torch import nn

from src.configs.schema import DecoderConfig, LatentConfig, SamplingConfig
from src.datasets.dataset import Events
from src.model.components.attention import MultiHeadAttention, ProjectedKeysValues
from src.model.components.conditioning import (
    UNCONDITIONED,
    AdaLNConditioning,
    LayerModulation,
    gate,
    modulate,
)
from src.model.components.embeddings import EventEmbeddings


@dataclass(frozen=True)
class SuffixCache:
    """The suffix self-attention's key/value cache, preallocated to its final length and filled
    one decode step at a time.

    `keys`/`values` are the full `max_steps` buffer; only the first `length` positions are
    written. Writing a step in place and handing back a view over it, rather than concatenating
    a newly-sized tensor onto the cache every step, is what keeps a step's cost independent of
    how far the generation has already run.
    """

    keys: torch.Tensor  # [batch_size, num_heads, max_steps, head_dim]
    values: torch.Tensor  # [batch_size, num_heads, max_steps, head_dim]
    length: int = 0

    def write(self, step: ProjectedKeysValues) -> 'SuffixCache':
        """Write one step's projection into the next free position, in place.

        Args:
            step: The new position's projection, `[batch_size, num_heads, 1, head_dim]`.
        Returns:
            The cache, one position longer. `keys`/`values` are the same buffer written into,
            not a copy.
        """
        self.keys[:, :, self.length : self.length + 1] = step.keys
        self.values[:, :, self.length : self.length + 1] = step.values
        return SuffixCache(keys=self.keys, values=self.values, length=self.length + 1)

    def filled(self) -> ProjectedKeysValues:
        """The positions written so far, as a view over the buffer rather than a copy."""
        return ProjectedKeysValues(
            keys=self.keys[:, :, : self.length], values=self.values[:, :, : self.length]
        )


@dataclass(frozen=True)
class LayerCache:
    """A KV cache for one decoder layer: the projected prefix, and the suffix positions already
    read."""

    prefix_kv: ProjectedKeysValues
    suffix_kv: SuffixCache


@dataclass(frozen=True)
class DecoderOutput:
    """What the decoder predicts at every suffix position: the activity to write there, the
    minutes until it, and the minutes until the case ends.

    Both times are a single standardized number on the scale their targets are encoded at, scored by
    `time_mae`. Neither head carries a spread of its own: what a cycle time could have been is the
    latent's to say where there is one, and the sampled activity path's where there is not. They
    overlap - a remaining time is the sum of the cycle times from that position on - and are read as
    two independent estimates rather than tied together.
    """

    activity_logits: torch.Tensor  # [batch_size, seq_len, num_activities]
    cycle_times: torch.Tensor  # [batch_size, seq_len], standardized
    remaining_times: torch.Tensor  # [batch_size, seq_len], standardized


@dataclass(frozen=True)
class GeneratedSuffix:
    """A batch of freely generated suffixes, kept as the raw predictions: EOT and everything
    after it included, with `lengths` saying where each suffix actually ended.

    The leading axes are whatever the caller generated over: `[batch_size, ...]` from
    `Decoder.generate`, `[batch_size, num_samples, ...]` from `TransformerCVAE.generate`.
    """

    activities: torch.Tensor  # [..., steps]
    lengths: torch.Tensor  # [...], events emitted before EOT, or `steps` if EOT never came
    cycle_times: torch.Tensor  # [..., steps], standardized like the targets
    # Read at position 0 alone, so it measures from the last prefix event, which is how the
    # reported remaining time is defined.
    remaining_time: torch.Tensor  # [...], standardized like the targets


class DecoderLayer(nn.Module):
    """One layer of the decoder stack, with self-attention over the suffix and cross-attention
    over the prefix."""

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
        # AdaLN conditioning supplies its own scale and shift, so the pre-norms only standardize.
        self.self_attention_norm = nn.LayerNorm(normalized_shape=d_model, elementwise_affine=False)
        self.cross_attention_norm = nn.LayerNorm(normalized_shape=d_model, elementwise_affine=False)
        self.feedforward_norm = nn.LayerNorm(normalized_shape=d_model, elementwise_affine=False)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        prefix_encoded: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        cache: LayerCache | None,
        modulation: LayerModulation,
    ) -> tuple[torch.Tensor, LayerCache]:
        """Run one layer over the suffix positions in `hidden`.

        Args:
            hidden: The positions to read, `[batch_size, seq_len, d_model]`: the whole suffix
                without a cache, the one event that follows it with one.
            prefix_encoded: The encoded prefix events, `[batch_size, prefix_seq_len, d_model]`.
            prefix_pad_mask: True where a prefix position holds padding.
            cache: What a previous call projected, or None to project everything.
            modulation: What the latent does to this layer, from `AdaLNConditioning.layers`.
        Returns:
            The layer's output for the positions read, and the cache to hand the next call.
        """
        hidden_norm = modulate(self.self_attention_norm(hidden), modulation.self_attention)
        step_kv = self.self_attention.project(hidden_norm)
        # With a cache, the new projection is written into the suffix positions already read.
        if cache is not None:
            suffix_cache = cache.suffix_kv.write(step_kv)
            suffix_kv = suffix_cache.filled()
        else:
            suffix_cache = SuffixCache(
                keys=step_kv.keys, values=step_kv.values, length=step_kv.keys.size(dim=2)
            )
            suffix_kv = step_kv
        hidden = hidden + gate(
            self.dropout(
                self.self_attention(query=hidden_norm, keys_values=suffix_kv, causal=cache is None)
            ),
            modulation.self_attention,
        )

        # The prefix is projected once, by `init_cache`, and read back here on every call after.
        prefix_kv = (
            cache.prefix_kv if cache is not None else self.cross_attention.project(prefix_encoded)
        )
        hidden = hidden + gate(
            self.dropout(
                self.cross_attention(
                    query=modulate(self.cross_attention_norm(hidden), modulation.cross_attention),
                    keys_values=prefix_kv,
                    key_padding_mask=prefix_pad_mask,
                )
            ),
            modulation.cross_attention,
        )

        hidden = hidden + gate(
            self.dropout(
                self.feedforward(modulate(self.feedforward_norm(hidden), modulation.feedforward))
            ),
            modulation.feedforward,
        )
        return hidden, LayerCache(prefix_kv=prefix_kv, suffix_kv=suffix_cache)

    def init_cache(self, prefix_encoded: torch.Tensor, *, max_steps: int) -> LayerCache:
        """Build this layer's cache for `generate`: the prefix projected once, and an empty
        suffix cache sized to `max_steps` for `forward` to write into one step at a time.

        Preallocating the suffix cache to its final size up front is what lets a step write
        into it in place, rather than reallocating and copying the whole cache the way
        concatenation would.

        Args:
            prefix_encoded: The encoded prefix events, `[batch_size, prefix_seq_len, d_model]`.
            max_steps: The suffix cache's capacity.
        Returns:
            This layer's starting cache, an empty suffix cache beside the projected prefix.
        """
        batch_size = prefix_encoded.size(dim=0)
        shape = (batch_size, self.self_attention.num_heads, max_steps, self.self_attention.head_dim)
        return LayerCache(
            prefix_kv=self.cross_attention.project(prefix_encoded),
            suffix_kv=SuffixCache(
                keys=prefix_encoded.new_zeros(size=shape),
                values=prefix_encoded.new_zeros(size=shape),
            ),
        )


def _nucleus(probabilities: torch.Tensor, *, top_p: float) -> torch.Tensor:
    """Keep the smallest set of activities whose probability sums to `top_p`, and renormalize.

    Where the temperature scales every step alike, this reads how peaked each one is: a step the
    process already determines leaves one activity above the threshold and the draw becomes the
    greedy read, where a genuine choice point keeps the activities that make it one. That is the
    whole reason it is here rather than a fixed candidate count, which cannot tell the two apart
    on a vocabulary this small.

    Args:
        probabilities: The activity head's softmax, `[batch_size, num_activities]`. Activities
            masked out upstream sit at zero and are sorted to the back, so they are never kept.
        top_p: The mass to keep. 1.0 keeps everything.
    Returns:
        The same shape, summing to 1 along the last dimension, zero outside the nucleus.
    """
    if top_p >= 1.0:
        return probabilities

    ordered, indices = probabilities.sort(dim=-1, descending=True)  # [batch_size, num_activities]
    # The mass strictly before each column. A column is kept when what came before it fell short
    # of the threshold, so the column that first reaches it is kept too and the heaviest activity
    # always is: the nucleus is never empty however peaked or flat the step.
    preceding = ordered.cumsum(dim=-1) - ordered  # [batch_size, num_activities]
    kept = ordered.masked_fill(mask=preceding >= top_p, value=0.0)

    # Back into vocabulary order, the dropped activities left at zero.
    probabilities = torch.zeros_like(input=probabilities).scatter(
        dim=-1, index=indices, src=kept
    )  # [batch_size, num_activities]
    return probabilities / probabilities.sum(dim=-1, keepdim=True)


class Decoder(nn.Module):
    """Transformer decoder that predicts a suffix of events, given the prefix and, where the model
    has one, a latent z.

    Applies self-attention over the suffix positions read so far, cross-attention over the
    encoded prefix. The latent z enters through `AdaLNConditioning`, which the decoder asks
    once per pass for what each layer is modulated by; with no latent every layer is handed
    `UNCONDITIONED` instead and the stack is the plain pre-norm decoder underneath.

    Where a model's variability lives decides how its activity head is read, which is what
    `sampling` says. A conditioned decoder declares none and is read greedily, one suffix per z: a
    whole suffix drawn from one latent is a coherent alternative continuation, where a per-step
    draw walks off the process's language while still looking locally plausible. A decoder with no
    latent has nowhere else to put its variability, so it declares a sampler and draws the activity
    from its logits at every step.

    The time heads are point regressors either way, and that is the whole of what they are: a single
    standardized number, scored by the absolute error. A head that emitted a scale of its own would
    be a second difference between the two arms, one the latent does not reach, and it would be paid
    for out of the activity head - a scale is free to shrink, and every step it shrinks by
    multiplies the gradient the shared trunk sees from the times against the one it sees from the
    activities. What a cycle time could have been is the latent's to carry where there is one. Where
    there is not, a cycle time still varies from draw to draw, because it is read off an activity
    path that was itself drawn; a remaining time, read at position 0 before any activity is written,
    does not, and that is a result about this arm rather than a gap in it.
    """

    def __init__(
        self,
        config: DecoderConfig,
        latent_config: LatentConfig | None,
        embeddings: EventEmbeddings,
        *,
        d_model: int,
        num_activities: int,
        sos_activity_index: int,
        pad_activity_index: int,
        pad_resource_index: int,
        eot_activity_index: int,
        sampling: SamplingConfig | None,
    ):
        """
        Args:
            config: The decoder's own hyperparameters.
            latent_config: The latent's width, or None for a decoder that reads no latent, whose
                layers go unmodulated.
            embeddings: The event embeddings, shared with the encoder.
            d_model: The width the stack runs at.
            num_activities: Rows of the activity head, the whole vocabulary.
            sos_activity_index: What the first decoder input is, in both teacher forcing and
                free-running generation.
            pad_activity_index: What `activity_dropout` blanks a teacher-forced activity to.
            pad_resource_index: The resource row every decoder input carries, the channel being
                one the decoder never feeds itself.
            eot_activity_index: What ends a generated suffix.
            sampling: How the activity logits are shaped before a draw reads them, or None for a
                decoder that reads them at their mode. Declaring one is what makes this decoder's
                heads the source of its variability, so it is also the whole of what tells the two
                arms apart here.
        """
        super().__init__()
        self.embeddings = embeddings
        self.dropout = nn.Dropout(p=config.dropout)
        # The embeddings are shared with the encoder but this norm is not: the two stacks read one
        # embedding space at whatever scale each of them settles on.
        self.embedding_norm = nn.LayerNorm(normalized_shape=d_model)
        self.activity_dropout = config.activity_dropout
        self.sampling = sampling

        self.sos_activity_index = sos_activity_index
        self.pad_activity_index = pad_activity_index
        self.pad_resource_index = pad_resource_index
        self.eot_activity_index = eot_activity_index

        self.conditioning = (
            AdaLNConditioning(
                latent_dim=latent_config.latent_dim, d_model=d_model, num_layers=config.num_layers
            )
            if latent_config is not None
            else None
        )
        self.layers = nn.ModuleList(
            DecoderLayer(config, d_model=d_model) for _ in range(config.num_layers)
        )
        # Pre-norm leaves the last layer's residual stream unnormalized, so the stack closes
        # with a norm of its own.
        self.norm = nn.LayerNorm(normalized_shape=d_model)

        # PAD opens no event and SOS only opens the sequence, so neither is an activity a suffix
        # can hold. A buffer rather than two ints, so it moves with the model and masks in one
        # call. EOT is not here: stopping is something the decoder has to be able to write. UNK is
        # not either, since a log's own suffixes carry it and it is part of what is being modelled.
        self.register_buffer(
            name='unemittable_activities',
            tensor=torch.tensor(data=[pad_activity_index, sos_activity_index], dtype=torch.long),
            persistent=False,
        )

        # A trunk shared by every head, so the heads can be smaller.
        self.shared_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout),
        )
        self.activity_head = nn.Linear(
            in_features=config.head_hidden_dim, out_features=num_activities
        )
        # One number each, the standardized time itself. Neither output is squashed: the targets
        # are standardized rather than bounded.
        self.cycle_time_head = nn.Linear(in_features=config.head_hidden_dim, out_features=1)
        self.remaining_time_head = nn.Linear(in_features=config.head_hidden_dim, out_features=1)

    def forward(
        self,
        suffix_activities: torch.Tensor,
        z: torch.Tensor | None,
        prefix_encoded: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
    ) -> DecoderOutput:
        """Predict an event for every position of a suffix at once.

        Args:
            suffix_activities: The ground-truth suffix activities, `[batch_size, seq_len]`. Read
                one step behind, so the decoder sees the truth up to each position rather than
                its own predictions.
            z: The sampled latent, `[batch_size, latent_dim]`, or None for a decoder built
                without one.
            prefix_encoded: The encoded prefix events, `[batch_size, prefix_seq_len, d_model]`.
            prefix_pad_mask: True where a prefix position holds padding.
        Returns:
            The per-position predictions.
        """
        decoder_input = self._teacher_forced_input(suffix_activities)
        if self.training and self.activity_dropout > 0.0:
            decoder_input = self._drop_activities(decoder_input)
        hidden, _ = self._run_layers(
            activities=decoder_input,
            prefix_encoded=prefix_encoded,
            prefix_pad_mask=prefix_pad_mask,
            start_position=0,
            caches=None,
            modulations=self._modulations(z),
        )  # [batch_size, seq_len, d_model]
        features = self.shared_layer(hidden)  # [batch_size, seq_len, head_hidden_dim]
        # Both time heads are [batch_size, seq_len, 1] -> [batch_size, seq_len].
        return DecoderOutput(
            activity_logits=self.activity_head(features),
            cycle_times=self.cycle_time_head(features).squeeze(dim=-1),
            remaining_times=self.remaining_time_head(features).squeeze(dim=-1),
        )

    def _modulations(self, z: torch.Tensor | None) -> tuple[LayerModulation, ...]:
        """What the latent does to each layer of the stack.

        Read once per pass rather than once per decode step, and answered without a latent at
        all where the model has none, so the stack below runs one code path either way.

        Args:
            z: The sampled latent, `[batch_size, latent_dim]`, or None.
        Returns:
            One modulation per layer, in stack order.
        """
        if self.conditioning is None:
            return (UNCONDITIONED,) * len(self.layers)
        return self.conditioning.layers(z)

    def read_with(self, sampling: SamplingConfig) -> None:
        """Replace the sampler the activity head is drawn through.

        Inference-time only: nothing here is a parameter or reaches the state dict, so swapping it
        changes how a trained model is read and not what it learned. That is what lets
        `pipelines.tune` search the pair over one set of weights rather than one run per value.

        Args:
            sampling: The sampler to draw with from here on.
        Raises:
            ValueError: If this decoder reads its activity head at its mode, there being nothing
                for a sampler to shape.
        """
        if self.sampling is None:
            raise ValueError(
                'this decoder reads its activity head at its mode, so there is no sampler to '
                'replace: its draws vary in z alone'
            )
        self.sampling = sampling

    def _next_activity(self, logits: torch.Tensor, *, drawing: bool) -> torch.Tensor:
        """Read the activity head for one decode step.

        The tokens no suffix can hold are masked out before either read. Greedily that changes
        nothing a trained model would have done, since it never ranks them first; drawing, it is
        the difference between a suffix and a decoding fault, because the head only ever learns to
        make PAD unlikely, never impossible, and a draw from the full softmax emits it at exactly
        whatever rate that leaves. Masking both reads rather than only the draw keeps the two
        architectures decoding under one constraint, which is what makes them comparable.

        A draw is then shaped by `self.sampling`: the temperature scales every step alike, the
        nucleus reads how peaked each one is. The masked tokens leave the softmax at zero, so
        neither knob can put them back.

        Args:
            logits: The head's output, `[batch_size, num_activities]`.
            drawing: Whether this step is a draw rather than the point prediction.
        Returns:
            The activity written at this position, `[batch_size]`.
        """
        logits = logits.index_fill(dim=-1, index=self.unemittable_activities, value=-torch.inf)
        if not drawing:
            return logits.argmax(dim=-1)
        # `self.sampling` is not None here: `drawing` is read off it in the first place.
        assert self.sampling is not None
        probabilities = (logits / self.sampling.temperature).softmax(dim=-1)
        probabilities = _nucleus(probabilities, top_p=self.sampling.top_p)
        return torch.multinomial(input=probabilities, num_samples=1).squeeze(dim=1)

    def _teacher_forced_input(self, suffix_activities: torch.Tensor) -> torch.Tensor:
        """What the decoder reads at each position: the suffix moved one step later, behind SOS.

        The same sequence `generate` feeds itself, which starts at SOS and appends what the
        previous step predicted. Positions past the suffix's length are masked out of the loss,
        so whatever the shift leaves there does not matter.

        Args:
            suffix_activities: The ground-truth suffix activities, `[batch_size, seq_len]`.
        Returns:
            The decoder input activities, the shape they came in at.
        """
        start = torch.full(
            size=(suffix_activities.size(dim=0), 1),
            fill_value=self.sos_activity_index,
            dtype=torch.long,
            device=suffix_activities.device,
        )
        return torch.cat(tensors=(start, suffix_activities[:, :-1]), dim=1)

    def _drop_activities(self, activities: torch.Tensor) -> torch.Tensor:
        """Blank a random `activity_dropout` fraction of the teacher-forced activities to PAD.

        A decoder that cannot count on the previous ground-truth token has to read what comes
        next off something else: z where there is one, which is what keeps information flowing
        through the latent, and the cross-attended prefix and the position being written where
        there is not. Either way it is what stops the previous token from being the whole answer,
        which is the state a free-running argmax read gets stuck in.
        """
        dropped = (torch.rand_like(activities, dtype=torch.float32) < self.activity_dropout) & (
            activities != self.sos_activity_index
        )  # [batch_size, seq_len]
        return activities.masked_fill(mask=dropped, value=self.pad_activity_index)

    def _run_layers(
        self,
        *,
        activities: torch.Tensor,
        prefix_encoded: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        start_position: int,
        caches: list[LayerCache] | None,
        modulations: tuple[LayerModulation, ...],
    ) -> tuple[torch.Tensor, list[LayerCache]]:
        """Embed a run of decoder inputs and push it through the stack.

        The one pass both teacher forcing and `generate` go through; `caches` is all that
        differs between them. Without one this reads a whole suffix under a causal mask, with
        one it reads the single event that follows what the caches already hold.

        Args:
            activities: The decoder input activities to read, `[batch_size, seq_len]`.
            prefix_encoded: The encoded prefix events, `[batch_size, prefix_seq_len, d_model]`.
            prefix_pad_mask: True where a prefix position holds padding.
            start_position: Where in the suffix `activities` starts, for the positional encoding.
            caches: One per layer, from a previous call, or None to read from the beginning.
            modulations: What the latent does to each layer, from `AdaLNConditioning.layers`.
        Returns:
            The stack's output for the positions read, and the caches carrying them.
        """
        hidden = self.embedding_norm(
            self.dropout(
                self.embeddings(self._blank_events(activities), start_position=start_position)
            )
        )  # [batch_size, seq_len, d_model]

        # No suffix padding mask: under the causal mask a padded position is visible only to
        # later positions, themselves padded and dropped from the loss. Masking it here would
        # leave a row with nothing to attend to, whose softmax is a NaN.
        layer_caches = caches if caches is not None else [None] * len(self.layers)
        new_caches: list[LayerCache] = []
        for layer, cache, modulation in zip(self.layers, layer_caches, modulations, strict=True):
            hidden, layer_cache = layer(
                hidden,
                prefix_encoded=prefix_encoded,
                prefix_pad_mask=prefix_pad_mask,
                cache=cache,
                modulation=modulation,
            )
            new_caches.append(layer_cache)

        return self.norm(hidden), new_caches

    def _blank_events(self, activities: torch.Tensor) -> Events:
        """Wrap decoder input activities as the events `EventEmbeddings` reads.

        The activity is the one channel the decoder feeds itself, so it is the one channel that
        may carry real content here: teacher forcing would otherwise hand it a ground-truth value
        that `generate` has only its own prediction of, and the two would read different things.
        Every other channel is blanked to the same PAD row or 0.0 scalar `generate` starts from,
        the predicted times included.

        Args:
            activities: Vocabulary indices to read as the activity channel, `[batch_size, seq_len]`.
        Returns:
            The events, ready for `EventEmbeddings`.
        """
        batch_size, seq_len = activities.shape
        device = activities.device
        return Events(
            activities=activities,
            resources=torch.full(
                size=(batch_size, seq_len),
                fill_value=self.pad_resource_index,
                dtype=torch.long,
                device=device,
            ),
            cycle_times=torch.zeros(size=(batch_size, seq_len), device=device),
            categorical_attributes=torch.zeros(
                size=(batch_size, seq_len, self.embeddings.num_categorical),
                dtype=torch.long,
                device=device,
            ),
            numeric_attributes=torch.zeros(
                size=(batch_size, seq_len, self.embeddings.num_numeric), device=device
            ),
            numeric_attributes_present=torch.zeros(
                size=(batch_size, seq_len, self.embeddings.num_numeric), device=device
            ),
            # Every position is read: the causal mask is what keeps a position from seeing
            # what follows it, and no caller asks these events for a padding mask.
            length=torch.full(
                size=(batch_size,), fill_value=seq_len, dtype=torch.long, device=device
            ),
        )

    def generate(
        self,
        z: torch.Tensor | None,
        prefix_encoded: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        max_steps: int,
        *,
        sample: bool,
    ) -> GeneratedSuffix:
        """Run the decoder free, feeding each step the event the previous one predicted.

        Every step is one cached call to `_run_layers`, the same pass teacher forcing runs, so
        writing n events costs n passes over one position.

        Only the activity head is ever drawn from, and only where `self.sampling` says so: a
        conditioned decoder declares none, so every step is greedy and two generations of one
        prefix differ in the z each was given. A decoder with no latent draws its activity at
        every step, and its point prediction is the same pass with `sample=False`. Both time
        heads are read as they are either way, so a cycle time varies from draw to draw only through
        the activity path it was read off.

        Args:
            z: The sampled latent, `[batch_size, latent_dim]`, or None for a decoder built
                without one.
            prefix_encoded: The encoded prefix events, `[batch_size, prefix_seq_len, d_model]`.
            prefix_pad_mask: True where a prefix position holds padding.
            max_steps: Hard cap on the suffix length, for generations that never emit EOT.
            sample: Whether this call is a draw rather than the point prediction. Nothing on a
                decoder whose activity head is not the source of its variability.
        Returns:
            The generated suffixes, the length of each, the minutes until each of their events,
            and the remaining time each was opened with.
        """
        batch_size = prefix_encoded.size(dim=0)
        device = prefix_encoded.device
        # The activity head is drawn from only where it is what the variability lives in;
        # everywhere else it is read at its mode, whatever the caller asked for.
        drawing = self.sampling is not None and sample
        # z does not change while a suffix is being written, so what it does to each layer is
        # read once here rather than at every step of the loop below.
        modulations = self._modulations(z)

        # What the decoder reads at each step: SOS first, exactly how `_teacher_forced_input`
        # opens, then the activity the previous step predicted.
        next_input = torch.full(
            size=(batch_size, 1),
            fill_value=self.sos_activity_index,
            dtype=torch.long,
            device=device,
        )

        generated_activities = torch.zeros(
            size=(batch_size, max_steps), dtype=torch.long, device=device
        )
        generated_cycle_times = torch.zeros(
            size=(batch_size, max_steps), dtype=prefix_encoded.dtype, device=device
        )
        # A row that never emits EOT ran to the cap, so that is the length it keeps.
        lengths = torch.full(
            size=(batch_size,), fill_value=max_steps, dtype=torch.long, device=device
        )
        finished = torch.zeros(size=(batch_size,), dtype=torch.bool, device=device)

        steps_taken = max_steps
        # Seeded before the loop rather than left None for its first iteration: every layer's
        # suffix cache is preallocated to `max_steps` right away, so even the first step writes
        # into it in place instead of starting the cache off at its exact size.
        caches: list[LayerCache] = [
            layer.init_cache(prefix_encoded, max_steps=max_steps) for layer in self.layers
        ]
        for position in range(max_steps):
            # Only this one position is new; everything before it is in `caches`.
            hidden, caches = self._run_layers(
                activities=next_input,
                prefix_encoded=prefix_encoded,
                prefix_pad_mask=prefix_pad_mask,
                start_position=position,
                caches=caches,
                modulations=modulations,
            )
            features = self.shared_layer(hidden[:, 0])  # [batch_size, head_hidden_dim]
            activities = self._next_activity(
                self.activity_head(features), drawing=drawing
            )  # [batch_size]
            # Only position 0, the state after SOS, answers for the whole suffix. Nothing has been
            # drawn yet there, so every sample of one prefix opens on the same remaining time.
            if position == 0:
                remaining_time = self.remaining_time_head(features).squeeze(dim=-1)  # [batch_size]

            generated_activities[:, position] = activities
            generated_cycle_times[:, position] = self.cycle_time_head(features).squeeze(dim=-1)
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
            cycle_times=generated_cycle_times[:, :steps_taken],  # [batch_size, steps]
            remaining_time=remaining_time,
        )
