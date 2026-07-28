from dataclasses import dataclass
import torch
from torch import nn

from src.configs.schema import AttentionConfig, DecoderConfig, LatentConfig
from src.model.components.attention import PrefixSuffixAttention
from src.model.components.embeddings import EventEmbeddings


@dataclass
class DecoderOutput:
    """What the decoder predicts for every suffix position."""
    activity_logits: torch.Tensor  # [batch_size, seq_len, num_activities]
    resource_logits: torch.Tensor  # [batch_size, seq_len, num_resources]
    timestamps: torch.Tensor       # [batch_size, seq_len], in [0, 1] like the targets


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
    ):
        super().__init__()
        self.embeddings = embeddings
        self.dropout = nn.Dropout(p=config.dropout)
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim

        # One projection for every layer's hidden and cell state at once, hence the 2 and the
        # `num_layers` factor; `initial_state` splits the result back apart.
        self.state_projection = nn.Linear(
            in_features=latent_config.latent_dim + prefix_dim,
            out_features=2 * config.num_layers * config.hidden_dim,
        )
        self.attention = PrefixSuffixAttention(
            attention_config, prefix_dim=prefix_dim, decoder_dim=config.hidden_dim
        )
        self.lstm = nn.LSTM(
            # Every step reads an embedded event, z (injection point 2 of 3) and the context the
            # previous step attended to.
            input_size=embeddings.output_dim + latent_config.latent_dim + self.attention.output_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            # nn.LSTM applies this between stacked layers, so a single-layer stack has nowhere
            # to put it. Passing it anyway is silently ignored, with a warning.
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
        output, (hidden, cell) = self.lstm(self.dropout(lstm_input), (state.hidden, state.cell))
        output = output.squeeze(dim=1)  # [batch_size, hidden_dim]

        # One projection reads this step's state as both the query and the memory row it leaves
        # behind. The row joins the memory before being attended over, so a step may read itself.
        query, memory_row = self.attention.project_state(output)
        suffix_memory = torch.cat(
            tensors=(state.suffix_memory, memory_row.unsqueeze(dim=1)), dim=1
        )  # [batch_size, steps_taken, attention_dim]
        context = self.attention(query, prefix_memory, suffix_memory, prefix_pad_mask)

        # z injection point 3 of 3. The trunk and the heads behind it are `predict`'s business,
        # so the loop over events carries no head-side work: what a step hands back is what it
        # holds, what it just read, and the latent it is decoding.
        representation = torch.cat(tensors=(output, context, z), dim=-1)

        return representation, DecoderState(
            hidden=hidden, cell=cell, context=context, suffix_memory=suffix_memory
        )

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
        batch: dict[str, torch.Tensor],
        z: torch.Tensor,
        prefix_outputs: torch.Tensor,
        prefix_summary: torch.Tensor,
    ) -> DecoderOutput:
        """
        Args:
            batch: A batch from `SuffixDataset`, read for its `decoder_input_*` and
                `prefix_len` entries.
            z: The sampled latent, `[batch_size, latent_dim]`.
            prefix_outputs: The prefix encoder's outputs, `[batch_size, seq_len, prefix_dim]`.
            prefix_summary: The prefix encoder's summary, `[batch_size, prefix_dim]`.
        Returns:
            The per-position predictions.
        """
        seq_len = batch['decoder_input_activities'].size(dim=1)

        # Teacher forcing: the step inputs are the ground-truth suffix shifted one behind SOS.
        # They are all known up front, so they are embedded in one call rather than per step.
        embedded = self.embeddings(
            batch['decoder_input_activities'],
            batch['decoder_input_resources'],
            batch['decoder_input_timestamps'],
        )  # [batch_size, seq_len, embeddings.output_dim]

        # The condition, prepared once: every step attends over the same projected prefix, and
        # the only thing ever hidden from it is that prefix's padding.
        prefix_memory = self.attention.prefix_projection(prefix_outputs)  # [batch_size, seq_len, attention_dim]
        # Built here, once, in the form attention consumes: every step masks the same padding,
        # and negating it inside the loop would pay for it `seq_len` times over.
        positions = torch.arange(end=seq_len, device=prefix_outputs.device)
        prefix_pad_mask = positions.unsqueeze(dim=0) >= batch['prefix_len'].unsqueeze(
            dim=1
        )  # [batch_size, seq_len]

        state = self.initial_state(z, prefix_summary)
        representations = []
        for position in range(seq_len):
            representation, state = self.step(
                embedded[:, position], z, state, prefix_memory, prefix_pad_mask
            )
            representations.append(representation)

        # One prediction per suffix position, for each field of an event. The stack is what lets
        # the trunk and the heads run once over the whole suffix instead of once per step.
        return self.predict(
            torch.stack(tensors=representations, dim=1)
        )  # [batch_size, seq_len, representation_dim]
