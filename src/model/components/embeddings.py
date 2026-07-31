import math

import torch
from torch import nn

from src.configs.dataset_info import DatasetInfo
from src.configs.schema import EmbeddingConfig
from src.datasets.dataset import EncodedEvents


class EventEmbeddings(nn.Module):
    """Turns a sequence of events into the vectors the transformer stacks read.
    An event is its activity encoding, its resource encoding and its scalar time delta,
    concatenated and projected to `d_model`. 
    
    A sinusoidal encoding of its position is added to the projected vector, so the stacks 
    can read the order of events out of the vectors.
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
        # +1 for the time delta, which is a scalar
        self.projection = nn.Linear(
            in_features=config.activity_dim + config.resource_dim + 1, out_features=d_model
        )
        # The embeddings are scaled up by the square root of the width.
        self.content_scale = math.sqrt(d_model)
        # Save the fixed positional encoding table as a buffer, so it moves with the model and is saved in checkpoints, but not trained.
        self.register_buffer(
            name='positional_encoding',
            tensor=_sinusoidal_encoding(length=dataset_info.max_trace_length, d_model=d_model),
            persistent=False,
        )
        self.output_dim = d_model

    def forward(self, events: EncodedEvents, *, start_position: int = 0) -> torch.Tensor:
        """
        Args:
            events: The events to embed, `[batch_size, seq_len]` per field. This is the one
                place the three are read apart, each having a table or a width of its own.
            start_position: Where in its sequence the first of them sits. A whole sequence
                starts at 0; a decoder generating with a cache hands over one event at a time
                and has to say which one, or every event would be encoded as the first.
        Returns:
            The embedded events, `[batch_size, seq_len, d_model]`.
        """
        # Concatenate the three fields of each event into a single vector, then project to `d_model`.
        event = torch.cat(
            tensors=(
                self.activity_embedding(events.activities),  # [batch_size, seq_len, activity_dim]
                self.resource_embedding(events.resources),   # [batch_size, seq_len, resource_dim]
                events.time_deltas.unsqueeze(dim=-1),        # [batch_size, seq_len, 1]
            ),
            dim=-1,
        )  # [batch_size, seq_len, activity_dim + resource_dim + 1]
        content = self.projection(event) * self.content_scale  # [batch_size, seq_len, d_model]

        # Add the fixed positional encoding to event vectors
        length = events.activities.size(dim=1)
        positions = self.positional_encoding[
            start_position : start_position + length
        ]  # [seq_len, d_model]
        return content + positions  # [batch_size, seq_len, d_model]


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
