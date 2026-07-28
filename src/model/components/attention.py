import math
import torch
from torch import nn
import torch.nn.functional as F

from src.configs.schema import AttentionConfig


class PrefixSuffixAttention(nn.Module):
    """Single-head scaled dot-product attention over the prefix and the suffix so far.

    The **query** is the current step's state; the **memory** is the concat of the prefix 
    and the suffix states produced so far.
    """

    def __init__(self, config: AttentionConfig, *, prefix_dim: int, decoder_dim: int):
        super().__init__()
        # Encoder outputs and decoder states come in at different widths; both become `dim`
        self.prefix_projection = nn.Linear(in_features=prefix_dim, out_features=config.dim)
        self.suffix_projection = nn.Linear(in_features=decoder_dim, out_features=config.dim)
        # The query is the same decoder state read through a separate projection, so what a
        # step asks for can differ from what it offers as memory
        self.query_projection = nn.Linear(in_features=decoder_dim, out_features=config.dim)
        self.dropout = nn.Dropout(p=config.dropout)
        self.scale = math.sqrt(config.dim)
        self.output_dim = config.dim

    def forward(
        self,
        state: torch.Tensor,
        prefix_memory: torch.Tensor,
        suffix_memory: torch.Tensor,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            state: The decoder state of the current step, `[batch_size, decoder_dim]`. It is
                the query; its projection is already the last row of `suffix_memory`.
            prefix_memory: The projected prefix outputs, `[batch_size, seq_len, dim]`.
            suffix_memory: The projected decoder states up to and including this step,
                `[batch_size, steps_taken, dim]`.
            prefix_mask: True where a prefix position holds a real event, `[batch_size, seq_len]`.
        Returns:
            The context vector, `[batch_size, dim]`.
        """
        # The query is the current step's state, projected to the attention width
        query = self.query_projection(state).unsqueeze(dim=1)  # [batch_size, 1, dim]

        # How much this step wants each memory position. Scaling by sqrt(dim) keeps the dot
        # products from growing with the width and saturating the softmax.
        prefix_scores = (
            query @ prefix_memory.transpose(dim0=-2, dim1=-1)
        ).squeeze(dim=1) / self.scale  # [batch_size, seq_len]
        suffix_scores = (
            query @ suffix_memory.transpose(dim0=-2, dim1=-1)
        ).squeeze(dim=1) / self.scale  # [batch_size, steps_taken]

        # Prefix padding is pushed to the dtype's minimum rather than dropped, which sends its
        # exponential to zero while leaving the row the same shape. The suffix doesn't need masking because the query is always the last row of the suffix memory
        prefix_scores = prefix_scores.masked_fill(
            mask=~prefix_mask, value=torch.finfo(prefix_scores.dtype).min
        )

        # The weights are the softmax over the concatenated prefix and suffix scores, so they sum to 1 across both halves
        weights = F.softmax(
            input=torch.cat(tensors=(prefix_scores, suffix_scores), dim=-1), dim=-1
        )

        # Dropout hits the weights, so a step is occasionally denied a position it wanted to read
        weights = self.dropout(weights).unsqueeze(dim=1)  # [batch_size, 1, seq_len + steps_taken]
        num_prefix = prefix_memory.size(dim=1)

        # The context is the weighted sum of the prefix and suffix memories, where the weights are the attention scores
        context = weights[..., :num_prefix] @ prefix_memory + weights[..., num_prefix:] @ suffix_memory
        return context.squeeze(dim=1)  # [batch_size, dim]
