import torch
from torch import nn

from src.configs.dataset_info import DatasetInfo
from src.configs.schema import EmbeddingConfig


class EventEmbeddings(nn.Module):
    """Turns a sequence of events into vectors. An event is the concatenation of its activity
    embedding, its resource embedding and its scalar time delta.

    One instance is shared by both trace encoders and the decoder, so an activity means the
    same vector wherever it is read or written.
    """

    def __init__(self, config: EmbeddingConfig, dataset_info: DatasetInfo):
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
        self.output_dim = config.activity_dim + config.resource_dim + 1

    def forward(
        self, activities: torch.Tensor, resources: torch.Tensor, time_deltas: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            activities: Activity indices, `[batch_size, seq_len]`.
            resources: Resource indices, `[batch_size, seq_len]`.
            time_deltas: Normalized time deltas, `[batch_size, seq_len]`.
        Returns:
            The embedded events, `[batch_size, seq_len, output_dim]`.
        """
        # One vector per event: the two looked-up embeddings with the time delta glued on.
        return torch.cat(
            tensors=(
                self.activity_embedding(activities),  # [batch_size, seq_len, activity_dim]
                self.resource_embedding(resources),   # [batch_size, seq_len, resource_dim]
                time_deltas.unsqueeze(dim=-1),        # [batch_size, seq_len, 1]
            ),
            dim=-1,
        )  # [batch_size, seq_len, output_dim]
