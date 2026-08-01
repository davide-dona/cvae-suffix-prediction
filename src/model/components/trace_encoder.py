import torch
from torch import nn

from src.configs.schema import TraceEncoderConfig
from src.datasets.dataset import EncodedEvents
from src.model.components.embeddings import EventEmbeddings


def padding_mask(lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Mark the positions of an encoder input that hold padding.

    True means padding, the polarity `nn.Transformer`'s `*_key_padding_mask` arguments take;
    the name says which way round it is.
    The extra leading column is the CLS token every encoder prepends, which is never padding.
    Args:
        lengths: Number of real events per sequence, `[batch_size]`.
        seq_len: The padded width the events come in at.
    Returns:
        `[batch_size, 1 + seq_len]`, True where the position holds padding.
    """
    positions = torch.arange(end=seq_len, device=lengths.device)
    events = positions.unsqueeze(dim=0) >= lengths.unsqueeze(dim=1)  # [batch_size, seq_len]
    cls_column = events.new_zeros(size=(lengths.size(dim=0), 1))     # [batch_size, 1]
    return torch.cat(tensors=(cls_column, events), dim=1)


class TraceEncoder(nn.Module):
    """Reads a padded sequence of events with a transformer encoder, every position attending
    over every other.

    A learned CLS token is prepended to the sequence, and the row it comes back as is the
    sequence's summary: the pooling is learned rather than fixed, so what a summary holds is
    whatever the latent networks reading it turn out to need.
    """
    def __init__(self, config: TraceEncoderConfig, embeddings: EventEmbeddings, *, d_model: int):
        super().__init__()
        self.embeddings = embeddings
        self.dropout = nn.Dropout(p=config.dropout)
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
                # leaves the residual path clean and is what lets the stack train without the
                # warmup schedule post-norm needs.
                norm_first=True,
            ),
            num_layers=config.num_layers,
            # Pre-norm leaves the last layer's residual stream unnormalized, so the stack closes
            # with a norm of its own.
            norm=nn.LayerNorm(normalized_shape=d_model),
            # The nested-tensor fast path does not apply to pre-norm layers. Saying so here is
            # what keeps it from being asked for and warned about once per encoder built.
            enable_nested_tensor=False,
        )
        self.output_dim = d_model

    def forward(self, events: EncodedEvents, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            events: The events to read, `[batch_size, seq_len]` per field.
            pad_mask: True where a position holds padding, `[batch_size, 1 + seq_len]`, from
                `padding_mask`.
        Returns:
            The encoded sequence, `[batch_size, 1 + seq_len, d_model]`, whose row 0 is the CLS
            summary and whose remaining rows are the events.
        """
        # One vector per event, positions included, then dropout so no single feature can carry
        # a position on its own.
        embedded = self.dropout(self.embeddings(events))  # [batch_size, seq_len, d_model]

        cls_token = self.cls_token.expand(embedded.size(dim=0), -1, -1)  # [batch_size, 1, d_model]
        sequence = torch.cat(tensors=(cls_token, embedded), dim=1)  # [batch_size, 1 + seq_len, d_model]

        # Masked positions are dropped from every attention row, so padding contributes nothing
        # to the summary or to any event's representation.
        return self.encoder(src=sequence, src_key_padding_mask=pad_mask)
