import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.configs.schema import TraceEncoderConfig
from src.model.components.embeddings import EventEmbeddings


class TraceEncoder(nn.Module):
    """Embeds a padded sequence of events with a bidirectional LSTM.
    
    
    Returns both the per-step outputs (what attention reads) and a fixed-width summary of the
    whole sequence (what the latent networks read).
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

        # One vector per event, then dropout so the LSTM cannot lean on a single feature.
        embedded = self.dropout(
            self.embeddings(activities, resources, timestamps)
        )  # [batch_size, seq_len, embeddings.output_dim]

        # Packing hides the padding from the LSTM: a padded step never updates the state, and
        # `final_hidden` therefore holds each sequence's last *real* step rather than its last
        # pad. Unpacking puts the rectangular shape back, with padded steps zeroed.
        packed = pack_padded_sequence(
            input=embedded, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, (final_hidden, _) = self.lstm(packed)  # [2 * num_layers, batch_size, hidden_dim]
        outputs, _ = pad_packed_sequence(
            sequence=packed_outputs, batch_first=True, total_length=seq_len
        )  # [batch_size, seq_len, output_dim]

        # The top layer's two directions: [-2] is the forward pass, which ends at the last real
        # event, and [-1] the backward one, which ends at the first. Together they summarize the
        # sequence read from both ends.
        summary = torch.cat(
            tensors=(final_hidden[-2], final_hidden[-1]), dim=-1
        )  # [batch_size, output_dim]
        return outputs, summary
