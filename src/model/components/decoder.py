from dataclasses import dataclass
import torch
from torch import nn

from src.configs.schema import AttentionConfig, DecoderConfig, LatentConfig
from src.datasets.dataset import PaddedEvents
from src.model.components.attention import PrefixSuffixAttention
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
    `[batch_size, num_samples, steps]` from `AttentionCVAE.generate`.
    """
    activities: torch.Tensor  # [..., steps]
    resources: torch.Tensor   # [..., steps]
    timestamps: torch.Tensor  # [..., steps], in [0, 1] like the targets
    lengths: torch.Tensor     # [...], events emitted before EOT, or `steps` if EOT never came


@dataclass
class DecoderState:
    """Everything one decoder step hands to the next."""
    hidden: torch.Tensor         # [num_layers, batch_size, hidden_dim]
    cell: torch.Tensor           # [num_layers, batch_size, hidden_dim]
    context: torch.Tensor        # [batch_size, attention_dim], what the last step attended to
    suffix_memory: torch.Tensor  # [batch_size, steps_taken, attention_dim], the projected states


class Decoder(nn.Module):
    """
    Generates the suffix one event at a time, attending over the prefix and its own past.

    A step attends after its recurrent update, then hands the context it read to the next
    step as part of its input. The prefix therefore reaches the recurrence itself rather than
    only the output heads, and stays visible at every position rather than only through the
    initial state.

    z is injected at three points, because attention gives the decoder a direct route to
    the prefix and a latent that only seeds the initial state is easy to ignore entirely:
        - it is mixed with the prefix summary into the initial hidden and cell states,
        - it is concatenated to every step's input,
        - it is concatenated into the shared layer behind the output heads.
    """

    def __init__(
        self,
        config: DecoderConfig,
        attention_config: AttentionConfig,
        latent_config: LatentConfig,
        embeddings: EventEmbeddings,
        *,
        prefix_dim: int,
        num_activities: int,
        num_resources: int,
        sos_activity_index: int,
        sos_resource_index: int,
        eot_activity_index: int,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.dropout = nn.Dropout(p=config.dropout)
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim

        # Only `generate` reads these: the token a suffix opens on, and the one it ends on.
        # Teacher forcing is handed both by the dataset instead, in `decoder_input_*`.
        self.sos_activity_index = sos_activity_index
        self.sos_resource_index = sos_resource_index
        self.eot_activity_index = eot_activity_index

        # One projection for every layer's hidden and cell state at once, hence the 2 and the
        # `num_layers` factor; `initial_state` splits the result back apart.
        self.state_projection = nn.Linear(
            in_features=latent_config.latent_dim + prefix_dim,
            out_features=2 * config.num_layers * config.hidden_dim,
        )
        self.attention = PrefixSuffixAttention(
            config=attention_config, prefix_dim=prefix_dim, decoder_dim=config.hidden_dim
        )
        self.lstm = nn.LSTM(
            # Every step reads an embedded event, z and the context the previous step attended to
            input_size=embeddings.output_dim + latent_config.latent_dim + self.attention.output_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )

        # A trunk shared by the three heads below, not a head itself: what an event is depends
        # on its activity, its resource and its timing together, so they are predicted off one
        # representation. Its input is the recurrent state, the attention context and z again.
        self.shared_layer = nn.Sequential(
            nn.Linear(
                in_features=config.hidden_dim + self.attention.output_dim + latent_config.latent_dim,
                out_features=config.head_hidden_dim,
            ),
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

    def initial_state(self, z: torch.Tensor, prefix_summary: torch.Tensor) -> DecoderState:
        """Open the recurrence for a batch of prefixes.

        Args:
            z: The sampled latent, `[batch_size, latent_dim]`.
            prefix_summary: The prefix encoder's summary, `[batch_size, prefix_dim]`.
        Returns:
            The state the first `step` continues from.
        """
        batch_size = z.size(dim=0)

        # z injection point 1 of 3: the recurrence starts already knowing both which prefix it
        # continues and which of that prefix's possible suffixes it was asked for. `tanh` puts
        # the states in the range an LSTM's own states live in.
        projected = self.state_projection(
            torch.cat(tensors=(z, prefix_summary), dim=-1)
        ).tanh()  # [batch_size, 2 * num_layers * hidden_dim]
        hidden, cell = projected.chunk(chunks=2, dim=-1)  # each [batch_size, num_layers * hidden_dim]
        # nn.LSTM wants the layer axis first, so both states become [num_layers, batch_size, hidden_dim].
        hidden = (
            hidden.view(batch_size, self.num_layers, self.hidden_dim)
            .transpose(dim0=0, dim1=1)
            .contiguous()
        )
        cell = (
            cell.view(batch_size, self.num_layers, self.hidden_dim)
            .transpose(dim0=0, dim1=1)
            .contiguous()
        )

        return DecoderState(
            hidden=hidden,
            cell=cell,
            # Nothing has been attended to yet, and no state exists to attend over.
            context=z.new_zeros(size=(batch_size, self.attention.output_dim)),
            suffix_memory=z.new_zeros(size=(batch_size, 0, self.attention.output_dim)),
        )

    def step(
        self,
        embedded_event: torch.Tensor,
        z: torch.Tensor,
        state: DecoderState,
        prefix_memory: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, DecoderState]:
        """Advance the recurrence by one event.

        Args:
            embedded_event: The event this step reads, `[batch_size, embeddings.output_dim]`.
                It is the ground-truth predecessor when teacher forcing and the model's own
                previous prediction when generating; the step itself cannot tell.
            z: The sampled latent, `[batch_size, latent_dim]`.
            state: What the previous step handed over, or `initial_state` at the first step.
            prefix_memory: The projected prefix outputs, `[batch_size, seq_len, attention_dim]`.
            prefix_pad_mask: True where a prefix position holds padding, `[batch_size, seq_len]`.
        Returns:
            This step's representation, `[batch_size, representation_dim]`, which `predict`
            reads an event out of, and the state the next step continues from.
        """
        # z injection point 2 of 3, alongside the context the previous step read: the latent is
        # an input at every step rather than a seed the recurrence may forget, and the prefix
        # informs the recurrence rather than only the heads hanging off it.
        lstm_input = torch.cat(
            tensors=(embedded_event, z, state.context), dim=-1
        ).unsqueeze(dim=1)
        # A length-1 sequence, so the layer stack and the dropout between layers stay nn.LSTM's
        # business while the loop over events is ours.
        output, (hidden, cell) = self.lstm(
            input=self.dropout(lstm_input), hx=(state.hidden, state.cell)
        )
        output = output.squeeze(dim=1)  # [batch_size, hidden_dim]

        # One projection reads this step's state as both the query and the memory row it leaves
        # behind. The row joins the memory before being attended over, so a step may read itself.
        query, memory_row = self.attention.project_state(output)
        suffix_memory = torch.cat(
            tensors=(state.suffix_memory, memory_row.unsqueeze(dim=1)), dim=1
        )  # [batch_size, steps_taken, attention_dim]
        context = self.attention(
            query=query,
            prefix_memory=prefix_memory,
            suffix_memory=suffix_memory,
            prefix_pad_mask=prefix_pad_mask,
        )

        # z injection point 3 of 3. The trunk and the heads behind it are `predict`'s business,
        # so the loop over events carries no head-side work: what a step hands back is what it
        # holds, what it just read, and the latent it is decoding.
        representation = torch.cat(tensors=(output, context, z), dim=-1)

        return representation, DecoderState(
            hidden=hidden, cell=cell, context=context, suffix_memory=suffix_memory
        )

    def _prepare_prefix(
        self, prefix_outputs: torch.Tensor, prefix_len: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Put the condition in the form every step consumes: the projected prefix, and the
        mask hiding its padding.

        Both are the same at every step, so they are built once per pass rather than inside the
        loop, and once here rather than once per path: teacher forcing and generation attend
        over the same prefix and must not disagree about how it was prepared.

        Args:
            prefix_outputs: The prefix encoder's outputs, `[batch_size, seq_len, prefix_dim]`.
            prefix_len: Number of real events per prefix, `[batch_size]`.
        Returns:
            The projected prefix memory, `[batch_size, seq_len, attention_dim]`, and a mask
            that is True where a prefix position holds padding, `[batch_size, seq_len]`.
        """
        prefix_memory = self.attention.prefix_projection(prefix_outputs)
        positions = torch.arange(end=prefix_outputs.size(dim=1), device=prefix_outputs.device)
        prefix_pad_mask = positions.unsqueeze(dim=0) >= prefix_len.unsqueeze(dim=1)
        return prefix_memory, prefix_pad_mask

    def predict(self, representation: torch.Tensor) -> DecoderOutput:
        """Read events out of the representations `step` produces.

        The trunk and the heads are position-wise, so this serves one step's representation,
        `[batch_size, representation_dim]`, as well as a whole suffix's stacked into
        `[batch_size, seq_len, representation_dim]`. Serving both is what lets the teacher-forced
        path keep the trunk out of its loop and run it once, over every position at once.
        """
        features = self.shared_layer(representation)  # [..., head_hidden_dim]
        return DecoderOutput(
            activity_logits=self.activity_head(features),
            resource_logits=self.resource_head(features),
            # Targets are min-max normalized into [0, 1], so the head is squashed to match.
            timestamps=self.timestamp_head(features).sigmoid().squeeze(dim=-1),
        )

    def forward(
        self,
        decoder_input: PaddedEvents,
        prefix_len: torch.Tensor,
        z: torch.Tensor,
        prefix_outputs: torch.Tensor,
        prefix_summary: torch.Tensor,
    ) -> DecoderOutput:
        """
        Args:
            decoder_input: The teacher-forced step inputs, the ground-truth suffix shifted one
                step behind SOS, `[batch_size, seq_len]` per field.
            prefix_len: Number of real events per prefix, `[batch_size]`.
            z: The sampled latent, `[batch_size, latent_dim]`.
            prefix_outputs: The prefix encoder's outputs, `[batch_size, seq_len, prefix_dim]`.
            prefix_summary: The prefix encoder's summary, `[batch_size, prefix_dim]`.
        Returns:
            The per-position predictions.
        """
        seq_len = decoder_input.activities.size(dim=1)

        # Teacher forcing: the step inputs are the ground-truth suffix shifted one behind SOS.
        # They are all known up front, so they are embedded in one call rather than per step.
        embedded = self.embeddings(
            activities=decoder_input.activities,
            resources=decoder_input.resources,
            timestamps=decoder_input.timestamps,
        )  # [batch_size, seq_len, embeddings.output_dim]

        prefix_memory, prefix_pad_mask = self._prepare_prefix(
            prefix_outputs=prefix_outputs, prefix_len=prefix_len
        )

        state = self.initial_state(z=z, prefix_summary=prefix_summary)
        representations = []
        for position in range(seq_len):
            representation, state = self.step(
                embedded_event=embedded[:, position],
                z=z,
                state=state,
                prefix_memory=prefix_memory,
                prefix_pad_mask=prefix_pad_mask,
            )
            representations.append(representation)

        # One prediction per suffix position, for each field of an event. The stack is what lets
        # the trunk and the heads run once over the whole suffix instead of once per step.
        return self.predict(
            torch.stack(tensors=representations, dim=1)
        )  # [batch_size, seq_len, representation_dim]

    def generate(
        self,
        z: torch.Tensor,
        prefix_outputs: torch.Tensor,
        prefix_summary: torch.Tensor,
        prefix_len: torch.Tensor,
        max_steps: int,
    ) -> GeneratedSuffix:
        """Run the decoder free, feeding each step the event the previous one predicted.

        The same `step` as `forward`, driven by the model's own output instead of a ground
        truth: the recurrence cannot tell the two apart, which is what keeps the path a suffix
        is generated on identical to the one it was trained on. The heads are read greedily, so
        the only thing that differs between two generations of one prefix is the z each was
        given, and a spread across them is a spread in `p(z | prefix)` rather than in a
        softmax sample.

        A row that has emitted EOT keeps being stepped until every row has, or `max_steps` is
        reached; `lengths` is what marks its ending, and the events past it are ignored rather
        than suppressed, which would cost a masked write in the per-step path for output no
        one reads.

        Args:
            z: The sampled latent, `[batch_size, latent_dim]`.
            prefix_outputs: The prefix encoder's outputs, `[batch_size, seq_len, prefix_dim]`.
            prefix_summary: The prefix encoder's summary, `[batch_size, prefix_dim]`.
            prefix_len: Number of real events per prefix, `[batch_size]`.
            max_steps: Hard cap on the suffix length, for the prefixes whose generation never
                emits EOT at all.
        Returns:
            The generated suffixes and the length of each.
        """
        batch_size = z.size(dim=0)
        prefix_memory, prefix_pad_mask = self._prepare_prefix(
            prefix_outputs=prefix_outputs, prefix_len=prefix_len
        )
        state = self.initial_state(z=z, prefix_summary=prefix_summary)

        # The first step reads SOS, exactly as the teacher-forced inputs open on it. Its
        # timestamp is 0.0, matching how `SuffixDataset` builds that position.
        activities = torch.full(
            size=(batch_size,), fill_value=self.sos_activity_index, dtype=torch.long, device=z.device
        )
        resources = torch.full(
            size=(batch_size,), fill_value=self.sos_resource_index, dtype=torch.long, device=z.device
        )
        timestamps = z.new_zeros(size=(batch_size,))

        # A row that never emits EOT ran to the cap, so that is the length it keeps.
        lengths = torch.full(
            size=(batch_size,), fill_value=max_steps, dtype=torch.long, device=z.device
        )
        finished = torch.zeros(size=(batch_size,), dtype=torch.bool, device=z.device)

        generated_activities, generated_resources, generated_timestamps = [], [], []
        for position in range(max_steps):
            representation, state = self.step(
                embedded_event=self.embeddings(
                    activities=activities, resources=resources, timestamps=timestamps
                ),
                z=z,
                state=state,
                prefix_memory=prefix_memory,
                prefix_pad_mask=prefix_pad_mask,
            )
            # One step's worth, so the trunk cannot be hoisted out of the loop the way the
            # teacher-forced path hoists it: nothing here knows the next step's input yet.
            prediction = self.predict(representation)

            activities = prediction.activity_logits.argmax(dim=-1)  # [batch_size]
            resources = prediction.resource_logits.argmax(dim=-1)   # [batch_size]
            timestamps = prediction.timestamps                      # [batch_size]

            generated_activities.append(activities)
            generated_resources.append(resources)
            generated_timestamps.append(timestamps)

            # A suffix ends at its first EOT, so a later one cannot move the length back.
            just_finished = ~finished & (activities == self.eot_activity_index)
            lengths = lengths.masked_fill(mask=just_finished, value=position)
            finished |= just_finished
            # Reading this stalls the device queue once per step, which the per-step path is
            # otherwise written to avoid. It pays for itself anyway: suffixes are far shorter
            # than `max_seq_len` on every log here, so most of the loop is skipped outright.
            if bool(finished.all()):
                break

        return GeneratedSuffix(
            activities=torch.stack(tensors=generated_activities, dim=1),  # [batch_size, steps]
            resources=torch.stack(tensors=generated_resources, dim=1),
            timestamps=torch.stack(tensors=generated_timestamps, dim=1),
            lengths=lengths,
        )
