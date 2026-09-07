from dataclasses import dataclass

import torch
from torch import nn

from src.configs.schema import TraceEncoderConfig
from src.datasets.dataset import Events
from src.model.components.embeddings import EventEmbeddings


@dataclass(frozen=True)
class EncodedTrace:
    """What `TraceEncoder` turns a sequence into: a summary of the whole, and a row per event."""

    summary: torch.Tensor  # [batch_size, d_model]
    events: torch.Tensor  # [batch_size, seq_len, d_model]


class TraceEncoder(nn.Module):
    """Reads a padded sequence of events with a transformer encoder, every position attending
    over every other.

    A learned CLS token is prepended to the sequence, and the row it comes back as is the
    sequence's summary: the pooling is learned rather than fixed, so what a summary holds is
    whatever the latent networks reading it turn out to need. The token is internal; callers
    see only the `EncodedTrace` it comes back as.
    """

    def __init__(self, config: TraceEncoderConfig, embeddings: EventEmbeddings, *, d_model: int):
        super().__init__()
        self.embeddings = embeddings
        self.dropout = nn.Dropout(p=config.dropout)
        # The embeddings are shared with the decoder but this norm is not: the two stacks read one
        # embedding space at whatever scale each of them settles on.
        self.embedding_norm = nn.LayerNorm(normalized_shape=d_model)
        # Initialize the CLS token to zero, so the encoder starts with no bias
        self.cls_token = nn.Parameter(data=torch.zeros(size=(1, 1, d_model)))

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=config.num_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                batch_first=True,
                # Pre-norm: each sublayer normalizes its input rather than its output, which
                # leaves the residual path clean. The run still warms its learning rate up, but
                # for the width of the batch rather than for the instability post-norm has.
                norm_first=True,
            ),
            num_layers=config.num_layers,
            # Pre-norm leaves the last layer's residual stream unnormalized, so the stack closes
            # with a norm of its own.
            norm=nn.LayerNorm(normalized_shape=d_model),
            enable_nested_tensor=False,
        )

    def forward(self, events: Events, pad_mask: torch.Tensor) -> EncodedTrace:
        """
        Args:
            events: The events to read, `[batch_size, seq_len]` per field.
            pad_mask: True where a position holds padding, `[batch_size, seq_len]`, from
                `Events.pad_mask`.
        Returns:
            The sequence's summary and its encoded events.
        """
        # Apply the embedding, dropout and norm to every event, then prepend the CLS token to
        # the sequence.
        embedded = self.embedding_norm(
            self.dropout(self.embeddings(events))
        )  # [batch_size, seq_len, d_model]

        cls_token = self.cls_token.expand(embedded.size(dim=0), -1, -1)  # [batch_size, 1, d_model]
        sequence = torch.cat(
            tensors=(cls_token, embedded), dim=1
        )  # [batch_size, 1 + seq_len, d_model]
        # The CLS column is never padding, so add a column of False to the pad mask to match
        # the sequence's width.
        cls_column = pad_mask.new_zeros(size=(pad_mask.size(dim=0), 1))  # [batch_size, 1]
        full_mask = torch.cat(tensors=(cls_column, pad_mask), dim=1)  # [batch_size, 1 + seq_len]

        # Masked positions are dropped from every attention row, so padding contributes nothing
        # to the summary or to any event's representation.
        encoded = self.encoder(
            src=sequence, src_key_padding_mask=full_mask
        )  # [batch_size, 1 + seq_len, d_model]
        return EncodedTrace(summary=encoded[:, 0], events=encoded[:, 1:])
