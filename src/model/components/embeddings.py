import math

import torch
from torch import nn

from src.configs.dataset_info import DatasetInfo
from src.configs.schema import EmbeddingConfig


class EventEmbeddings(nn.Module):
    """Turns a sequence of events into the vectors the transformer stacks read.

    An event is its activity embedding, its resource embedding and its scalar time delta,
    concatenated and projected to `d_model`, with a sinusoidal encoding of its position added
    on. Attention is order-blind, so the position has to be carried by the vector itself.

    One instance is shared by both trace encoders and the decoder, so an activity means the
    same vector wherever it is read or written, and every one of the three indexes positions
    from the start of its own sequence.
    """

    def __init__(self, config: EmbeddingConfig, dataset_info: DatasetInfo, *, d_model: int):
        super().__init__()
        # `padding_idx` pins the PAD row to zero and keeps it out of the gradient, so padded
        # steps contribute nothing to what the embedding learns.
        self.activity_embedding = nn.Embedding(
            num_embeddings=dataset_info.num_activities,
            embedding_dim=config.activity_dim,
            padding_idx=dataset_info.pad_activity_index,
        )
        self.resource_embedding = nn.Embedding(
            num_embeddings=dataset_info.num_resources,
            embedding_dim=config.resource_dim,
            padding_idx=dataset_info.pad_resource_index,
        )
        # The trailing 1 is the time delta, which is a scalar and needs no embedding table.
        # Activity and resource keep widths of their own, sized to their vocabularies, and the
        # projection is what brings an event up to the width the stacks run at.
        self.projection = nn.Linear(
            in_features=config.activity_dim + config.resource_dim + 1, out_features=d_model
        )
        # Not persistent: it is a function of `max_trace_length` and `d_model`, both of which are
        # known at build time, so storing it in every checkpoint would only be a way for the
        # saved table and the rebuilt model to disagree.
        self.register_buffer(
            name='positional_encoding',
            tensor=_sinusoidal_encoding(length=dataset_info.max_trace_length, d_model=d_model),
            persistent=False,
        )
        self.output_dim = d_model

    def forward(
        self, activities: torch.Tensor, resources: torch.Tensor, timestamps: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            activities: Activity indices, `[batch_size, seq_len]`.
            resources: Resource indices, `[batch_size, seq_len]`.
            timestamps: Normalized time deltas, `[batch_size, seq_len]`.
        Returns:
            The embedded events, `[batch_size, seq_len, d_model]`.
        """
        # One vector per event: the two looked-up embeddings with the time delta glued on.
        event = torch.cat(
            tensors=(
                self.activity_embedding(activities),  # [batch_size, seq_len, activity_dim]
                self.resource_embedding(resources),   # [batch_size, seq_len, resource_dim]
                timestamps.unsqueeze(dim=-1),         # [batch_size, seq_len, 1]
            ),
            dim=-1,
        )  # [batch_size, seq_len, activity_dim + resource_dim + 1]

        # The leading `seq_len` rows of the table, broadcast over the batch: position `i` of every
        # sequence in the batch is the i-th event of that sequence.
        positions = self.positional_encoding[: activities.size(dim=1)]  # [seq_len, d_model]
        return self.projection(event) + positions  # [batch_size, seq_len, d_model]


def _sinusoidal_encoding(length: int, d_model: int) -> torch.Tensor:
    """Build the fixed position table, `[length, d_model]`.

    Each pair of channels holds a sine and a cosine of the position at a different frequency, the
    wavelengths running from 2pi up to 10000 * 2pi. A position is therefore a distinct vector, and
    a fixed offset between two positions is a fixed linear map between their encodings, which is
    what lets attention read relative order out of them.

    Args:
        length: How many positions to build, i.e. the longest sequence the model will see.
        d_model: The width to build them at; an odd width leaves the last channel a sine.
    Returns:
        The table, ready to be added to a `[..., length, d_model]` batch of embedded events.
    """
    positions = torch.arange(end=length, dtype=torch.float32).unsqueeze(dim=1)  # [length, 1]
    # exp(-log(10000) * 2i / d_model) is 10000^(-2i/d_model) computed in log space, which keeps
    # the smallest frequency from underflowing at wide `d_model`.
    frequencies = torch.exp(
        input=torch.arange(start=0, end=d_model, step=2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )  # [ceil(d_model / 2)]

    encoding = torch.zeros(size=(length, d_model), dtype=torch.float32)
    angles = positions * frequencies  # [length, ceil(d_model / 2)]
    encoding[:, 0::2] = torch.sin(input=angles)
    # An odd `d_model` leaves the cosine half one channel short of the sine half.
    encoding[:, 1::2] = torch.cos(input=angles[:, : d_model // 2])
    return encoding
