import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from src.configs.schema import TraceEncoderConfig
from src.model.components.embeddings import EventEmbeddings


class TraceEncoder(nn.Module):
    """Embeds a padded sequence of events with a bidirectional LSTM.

    Two ways in, over one recurrence: `forward` returns both the per-step outputs (what
    attention reads) and a fixed-width summary of the whole sequence (what the latent networks
    read), while `summarize` returns the summary alone and never unpacks the outputs, for the
    callers that have nothing to feed them to.
    """

    def __init__(self, config: TraceEncoderConfig, embeddings: EventEmbeddings):
        super().__init__()
        self.embeddings = embeddings
        self.dropout = nn.Dropout(p=config.dropout)
        self.lstm = nn.LSTM(
            input_size=embeddings.output_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            # nn.LSTM applies this between stacked layers, so a single-layer stack has nowhere
            # to put it. Passing it anyway is silently ignored, with a warning.
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        # Twice the hidden size, because forward and backward states are concatenated.
        self.output_dim = config.hidden_dim * 2

    def _read(
        self,
        activities: torch.Tensor,
        resources: torch.Tensor,
        timestamps: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[PackedSequence, torch.Tensor]:
        """Run the LSTM over a padded batch, in the packed form it produces.

        The whole recurrence lives here, so `forward` and `summarize` differ only in whether
        they go on to materialize the per-step outputs.

        Args:
            activities, resources, timestamps: The events to read, `[batch_size, seq_len]` each.
            lengths: Number of real events per sequence, `[batch_size]`.
        Returns:
            The packed per-step outputs, and the sequence summary `[batch_size, output_dim]`.
        """
        # One vector per event, then dropout so the LSTM cannot lean on a single feature.
        embedded = self.dropout(
            self.embeddings(activities=activities, resources=resources, timestamps=timestamps)
        )  # [batch_size, seq_len, embeddings.output_dim]

        # Packing hides the padding from the LSTM: a padded step never updates the state, and
        # `final_hidden` therefore holds each sequence's last *real* step rather than its last pad.
        packed = pack_padded_sequence(
            input=embedded, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, (final_hidden, _) = self.lstm(input=packed)  # [2 * num_layers, batch_size, hidden_dim]

        # The top layer's two directions: [-2] is the forward pass, which ends at the last real
        # event, and [-1] the backward one, which ends at the first. Together they summarize the
        # sequence read from both ends.
        summary = torch.cat(
            tensors=(final_hidden[-2], final_hidden[-1]), dim=-1
        )  # [batch_size, output_dim]
        return packed_outputs, summary

    def summarize(
        self,
        activities: torch.Tensor,
        resources: torch.Tensor,
        timestamps: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Read a sequence for its summary alone.

        The summary falls out of the LSTM's final hidden state, so the per-step outputs are
        never unpacked: for a caller that only wants the summary, the rectangular
        `[batch_size, seq_len, output_dim]` tensor would be allocated and scattered into only
        to be dropped. That is the suffix encoder's whole use, and it runs once per batch.

        Args:
            activities, resources, timestamps: The events to read, `[batch_size, seq_len]` each.
            lengths: Number of real events per sequence, `[batch_size]`.
        Returns:
            The sequence summary, `[batch_size, output_dim]`.
        """
        _, summary = self._read(activities, resources, timestamps, lengths)
        return summary

    def forward(
        self,
        activities: torch.Tensor,
        resources: torch.Tensor,
        timestamps: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            activities, resources, timestamps: The events to read, `[batch_size, seq_len]` each.
            lengths: Number of real events per sequence, `[batch_size]`.
        Returns:
            The per-step outputs `[batch_size, seq_len, output_dim]` (padded positions are
            zero), and the sequence summary `[batch_size, output_dim]`.
        """
        # The padded length, kept so unpacking restores exactly the shape that came in.
        seq_len = activities.size(dim=1)

        packed_outputs, summary = self._read(activities, resources, timestamps, lengths)
        # Unpacking puts the rectangular shape back, with padded steps zeroed.
        outputs, _ = pad_packed_sequence(
            sequence=packed_outputs, batch_first=True, total_length=seq_len
        )  # [batch_size, seq_len, output_dim]
        return outputs, summary
