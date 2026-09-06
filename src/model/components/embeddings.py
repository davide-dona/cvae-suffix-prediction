import math

import torch
from torch import nn

from src.configs.schema import EmbeddingConfig
from src.datasets.codec import DatasetCodec
from src.datasets.dataset import Events


class EventEmbeddings(nn.Module):
    """Turns a sequence of events into the vectors the transformer stacks read.
    An event is its activity encoding, its resource encoding, the minutes of cycle time it took and
    whatever channels `data.event_features` offers, concatenated and projected to `d_model`.

    A sinusoidal encoding of its position is added to the projected vector, so the stacks
    can read the order of events out of the vectors. The stacks normalize what comes back before
    reading it; nothing here does, so that the encoder's norm and the decoder's stay their own.
    """

    def __init__(self, config: EmbeddingConfig, codec: DatasetCodec, *, d_model: int):
        super().__init__()
        self.activity_embedding = nn.Embedding(
            num_embeddings=codec.activity.num_rows,
            embedding_dim=config.activity_dim,
            padding_idx=codec.activity.pad_index,
        )
        self.resource_embedding = nn.Embedding(
            num_embeddings=codec.resource.num_rows,
            embedding_dim=config.resource_dim,
            padding_idx=codec.resource.pad_index,
        )
        self.num_categorical = len(codec.categorical_features)
        self.num_numeric = len(codec.numeric_features)

        self.feature_embedding = (
            nn.Embedding(
                num_embeddings=codec.num_feature_categories,
                embedding_dim=config.feature_dim,
                padding_idx=0,
            )
            if self.num_categorical > 0
            else None
        )
        # Two scalars per numeric attribute (value + presence); the cycle time before an event is
        # never missing, so it gets one.
        self.projection = nn.Linear(
            in_features=(
                config.activity_dim
                + config.resource_dim
                + 1
                + self.num_categorical * config.feature_dim
                + 2 * self.num_numeric
            ),
            out_features=d_model,
        )
        # A buffer moves with the model but is not trained.
        self.register_buffer(
            name='positional_encoding',
            tensor=_sinusoidal_encoding(length=codec.max_trace_length, d_model=d_model),
            persistent=False,
        )

    def forward(self, events: Events, *, start_position: int = 0) -> torch.Tensor:
        """
        Args:
            events: The events to embed, `[batch_size, seq_len]` per field. This is the one
                place the channels are read apart, each having a table or a width of its own.
            start_position: Where in its sequence the first of them sits. A whole sequence
                starts at 0; a decoder generating with a cache hands over one event at a time
                and has to say which one, or every event would be encoded as the first.
        Returns:
            The embedded events, `[batch_size, seq_len, d_model]`.
        """
        # Concatenate every channel of each event into a single vector, then project to `d_model`.
        channels = [
            self.activity_embedding(events.activities),  # [batch_size, seq_len, activity_dim]
            self.resource_embedding(events.resources),  # [batch_size, seq_len, resource_dim]
            events.cycle_times.unsqueeze(dim=-1),  # [batch_size, seq_len, 1]
        ]
        if self.feature_embedding is not None:
            channels.append(
                self.feature_embedding(events.categorical_attributes).flatten(start_dim=-2)
            )  # [batch_size, seq_len, num_categorical * feature_dim]
        # Zero-width on a log with no numeric attributes, which `cat` takes as the no-op it is.
        channels.append(events.numeric_attributes)  # [batch_size, seq_len, num_numeric]
        channels.append(events.numeric_attributes_present)  # [batch_size, seq_len, num_numeric]

        # Concatenate the channels and project to `d_model`.
        event = torch.cat(tensors=channels, dim=-1)  # [batch_size, seq_len, projection.in_features]
        content = self.projection(event)  # [batch_size, seq_len, d_model]

        # Add the fixed positional encoding to event vectors
        length = events.activities.size(dim=1)
        positions = self.positional_encoding[
            start_position : start_position + length
        ]  # [seq_len, d_model]
        return content + positions  # [batch_size, seq_len, d_model]


def _sinusoidal_encoding(length: int, d_model: int) -> torch.Tensor:
    """Build the fixed position table, `[length, d_model]`.
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
