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
        # A decoder state is read two ways: as the query this step asks with, and as the memory
        # row later steps attend over. They stay separate projections - what a step asks for can
        # differ from what it offers - but they are one matmul, split apart by `project_state`,
        # because this is the per-step path and every op in it is paid `seq_len` times.
        self.state_projection = nn.Linear(in_features=decoder_dim, out_features=2 * config.dim)
        self.dropout = nn.Dropout(p=config.dropout)
        self.scale = math.sqrt(config.dim)
        self.output_dim = config.dim

    def project_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Read a decoder state as both a query and a memory row.

        Args:
            state: The decoder state of a step, `[batch_size, decoder_dim]`.
        Returns:
            The query that step attends with and the memory row it leaves behind for later
            steps, `[batch_size, dim]` each.
        """
        query, memory = self.state_projection(state).chunk(chunks=2, dim=-1)
        # Scaling by sqrt(dim) keeps the dot products from growing with the width and saturating
        # the softmax. Doing it to the query costs one op; doing it to the scores costs two.
        return query / self.scale, memory

    def forward(
        self,
        query: torch.Tensor,
        prefix_memory: torch.Tensor,
        suffix_memory: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: What this step attends with, `[batch_size, dim]`, already scaled by
                `project_state`. Its memory half is already the last row of `suffix_memory`.
            prefix_memory: The projected prefix outputs, `[batch_size, seq_len, dim]`.
            suffix_memory: The projected decoder states up to and including this step,
                `[batch_size, steps_taken, dim]`.
            prefix_pad_mask: True where a prefix position holds padding, `[batch_size, seq_len]`.
        Returns:
            The context vector, `[batch_size, dim]`.
        """
        # How much this step wants each memory position. Scoring as `memory @ query` rather than
        # `query @ memory.T` leaves both memories in the layout they are already stored in: the
        # transposed form makes the backend materialize a transposed copy of the memory once per
        # step, which is most of what the per-step path costs on a GPU.
        query = query.unsqueeze(dim=-1)  # [batch_size, dim, 1]
        prefix_scores = (prefix_memory @ query).squeeze(dim=-1)  # [batch_size, seq_len]
        suffix_scores = (suffix_memory @ query).squeeze(dim=-1)  # [batch_size, steps_taken]

        # Prefix padding is pushed to the dtype's minimum rather than dropped, which sends its
        # exponential to zero while leaving the row the same shape. The suffix doesn't need masking because the query is always the last row of the suffix memory
        prefix_scores = prefix_scores.masked_fill(
            mask=prefix_pad_mask, value=torch.finfo(prefix_scores.dtype).min
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
