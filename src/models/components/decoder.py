from dataclasses import dataclass
import torch
from torch import nn

from src.configs.schema import AttentionConfig, DecoderConfig, LatentConfig
from src.models.components.attention import PrefixSuffixAttention, build_attention_mask
from src.models.components.embeddings import EventEmbeddings


@dataclass
class DecoderOutput:
    """What the decoder predicts for every suffix position."""
    activity_logits: torch.Tensor  # [batch_size, seq_len, num_activities]
    resource_logits: torch.Tensor  # [batch_size, seq_len, num_resources]
    timestamps: torch.Tensor       # [batch_size, seq_len], in [0, 1] like the targets


class Decoder(nn.Module):
    """
    Generates the suffix, one event per step, attending over the prefix and its own past.

    The recurrence runs over the whole teacher-forced suffix in one call and attention is
    applied to the resulting states afterwards. Attention therefore informs the output
    heads rather than the recurrence itself, which is what makes a single batched pass
    possible; the causal mask is what keeps that pass honest.

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
        self.dropout = nn.Dropout(config.dropout)
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim

        # One projection for every layer's hidden and cell state at once, hence the 2 and the
        # `num_layers` factor; `forward` splits the result back apart.
        self.initial_state = nn.Linear(
            in_features=latent_config.latent_dim + prefix_dim, out_features=2 * config.num_layers * config.hidden_dim
        )
        self.lstm = nn.LSTM(
            # Every step reads an embedded event with z appended (z injection point 2 of 3).
            input_size=embeddings.output_dim + latent_config.latent_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            batch_first=True,
        )
        self.attention = PrefixSuffixAttention(
            attention_config, prefix_dim=prefix_dim, decoder_dim=config.hidden_dim
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
            nn.Dropout(config.dropout),
        )
        # One head per field of an event; the timestamp is a scalar, so its head is width 1.
        self.activity_head = nn.Linear(config.head_hidden_dim, num_activities)
        self.resource_head = nn.Linear(config.head_hidden_dim, num_resources)
        self.timestamp_head = nn.Linear(config.head_hidden_dim, 1)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        z: torch.Tensor,
        prefix_outputs: torch.Tensor,
        prefix_summary: torch.Tensor,
    ) -> DecoderOutput:
        """
        Args:
            batch: A batch from `SuffixDataset`, read for its `decoder_input_*`, `prefix_len`
                and `suffix_len` entries.
            z: The sampled latent, `[batch_size, latent_dim]`.
            prefix_outputs: The prefix encoder's outputs, `[batch_size, seq_len, prefix_dim]`.
            prefix_summary: The prefix encoder's summary, `[batch_size, prefix_dim]`.
        Returns:
            The per-position predictions.
        """
        batch_size, seq_len = batch['decoder_input_activities'].shape

        # z injection point 1 of 3: the recurrence starts already knowing both which prefix it
        # continues and which of that prefix's possible suffixes it was asked for. `tanh` puts
        # the states in the range an LSTM's own states live in.
        initial = self.initial_state(
            torch.cat((z, prefix_summary), dim=-1)
        ).tanh()  # [batch_size, 2 * num_layers * hidden_dim]
        hidden, cell = initial.chunk(2, dim=-1)  # each [batch_size, num_layers * hidden_dim]
        # nn.LSTM wants the layer axis first, so both states become [num_layers, batch_size, hidden_dim].
        hidden = hidden.view(batch_size, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        cell = cell.view(batch_size, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()

        # Teacher forcing: the step inputs are the ground-truth suffix shifted one behind SOS, so
        # the whole recurrence is one batched pass instead of `seq_len` single-step calls.
        embedded = self.embeddings(
            batch['decoder_input_activities'],
            batch['decoder_input_resources'],
            batch['decoder_input_timestamps'],
        )  # [batch_size, seq_len, embeddings.output_dim]
        # z injection point 2 of 3: the same latent repeated, so it is an input at every step
        # rather than a seed the recurrence is free to forget after the first one.
        z_per_step = z.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, latent_dim]
        lstm_input = torch.cat((embedded, z_per_step), dim=-1)  # [batch_size, seq_len, embed + latent]
        states, _ = self.lstm(self.dropout(lstm_input), (hidden, cell))  # [batch_size, seq_len, hidden_dim]

        # Attention comes after the recurrence, over the finished states: every step reads the
        # prefix and its own past, and the causal half of the mask is what keeps a step from
        # reading states the recurrence had not produced yet at that point.
        mask = build_attention_mask(
            batch['prefix_len'], batch['suffix_len'], seq_len
        )  # [batch_size, seq_len, 2 * seq_len]
        context = self.attention(states, prefix_outputs, mask)  # [batch_size, seq_len, attention.output_dim]

        # z injection point 3 of 3, into the trunk the three heads read from.
        features = self.shared_layer(
            torch.cat((states, context, z_per_step), dim=-1)
        )  # [batch_size, seq_len, head_hidden_dim]

        # One prediction per suffix position, for each field of an event.
        return DecoderOutput(
            activity_logits=self.activity_head(features),  # [batch_size, seq_len, num_activities]
            resource_logits=self.resource_head(features),  # [batch_size, seq_len, num_resources]
            # Targets are min-max normalized into [0, 1], so the head is squashed to match.
            timestamps=self.timestamp_head(features).sigmoid().squeeze(-1),  # [batch_size, seq_len]
        )
