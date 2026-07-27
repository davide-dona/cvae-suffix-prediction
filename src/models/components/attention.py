import math
import torch
from torch import nn
import torch.nn.functional as F

from src.configs.schema import AttentionConfig


def build_attention_mask(
    prefix_len: torch.Tensor, suffix_len: torch.Tensor, seq_len: int
) -> torch.Tensor:
    """
    Build the boolean mask saying which memory positions each decoder step may read.

    The memory is the prefix followed by the suffix, so the mask has two halves:
    - prefix columns are visible to every step, up to that sequence's `prefix_len`. 
      The whole prefix is observed, so there is nothing to hide.
    - suffix columns are visible only up to and including the current step. This
      is the causal mask that prevents the decoder from cheating by reading its own future.
    Args:
        prefix_len: Real prefix length per sequence, `[batch_size]`.
        suffix_len: Real suffix length per sequence, `[batch_size]`.
        seq_len: The padded length both prefix and suffix are stored at.
    Returns:
        A boolean mask `[batch_size, seq_len, 2 * seq_len]`, True where reading is allowed.
    """
    # Compared against the lengths below, this is what turns a count into a per-position test.
    positions = torch.arange(seq_len, device=prefix_len.device)  # [seq_len]

    # Prefix half: open up to each sequence's real length, so only its padding stays closed.
    prefix_mask = positions.unsqueeze(0) < prefix_len.unsqueeze(1)  # [batch_size, seq_len (key)]
    # The same row is handed to every decoder step, since none of them has anything to hide.
    prefix_mask = prefix_mask.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, query, key]

    # Suffix half: a step may look at itself and everything behind it, but not ahead ...
    causal = positions.unsqueeze(0) <= positions.unsqueeze(1)         # [seq_len (query), seq_len (key)]
    # ... and never past the real suffix, so padded steps are closed for every query.
    within_suffix = positions.unsqueeze(0) < suffix_len.unsqueeze(1)  # [batch_size, seq_len (key)]
    # Broadcast of [1, query, key] against [batch_size, 1, key]: both conditions must hold.
    suffix_mask = causal.unsqueeze(0) & within_suffix.unsqueeze(1)    # [batch_size, seq_len, seq_len]

    # The halves sit side by side in the same order the memory is concatenated in.
    return torch.cat((prefix_mask, suffix_mask), dim=-1)  # [batch_size, seq_len, 2 * seq_len]


class PrefixSuffixAttention(nn.Module):
    """Single-head scaled dot-product attention over the prefix and the suffix so far.

    The two memories arrive at different widths and are brought to a shared one by a linear
    layer each, which is also what lets them be concatenated into a single memory.

    Every decoder step is scored in a single matmul rather than one step at a time, which is
    why the decoder runs its recurrence first and attends afterwards.
    """

    def __init__(self, config: AttentionConfig, *, prefix_dim: int, decoder_dim: int):
        super().__init__()
        # Encoder outputs and decoder states come in at different widths; both become `dim`.
        self.prefix_projection = nn.Linear(prefix_dim, config.dim)
        self.suffix_projection = nn.Linear(decoder_dim, config.dim)
        # The queries are the same decoder states read through a separate projection, so what a
        # step asks for can differ from what it offers as memory.
        self.query_projection = nn.Linear(decoder_dim, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.scale = math.sqrt(config.dim)
        self.output_dim = config.dim

    def forward(
        self, decoder_states: torch.Tensor, prefix_outputs: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            decoder_states: The decoder's per-step states, `[batch_size, seq_len, decoder_dim]`.
                These are both the queries and the suffix half of the memory.
            prefix_outputs: The prefix encoder's outputs, `[batch_size, seq_len, prefix_dim]`.
            mask: The mask from `build_attention_mask`, `[batch_size, seq_len, 2 * seq_len]`.
        Returns:
            The context vectors, `[batch_size, seq_len, dim]`.
        """
        # The prefix events followed by the decoder's own states: one memory a step can read
        # both from, in the order `build_attention_mask` assumes.
        memory = torch.cat(
            (self.prefix_projection(prefix_outputs), self.suffix_projection(decoder_states)), dim=1
        )  # [batch_size, 2 * seq_len, dim]
        queries = self.query_projection(decoder_states)  # [batch_size, seq_len, dim]

        # scores[b, i, j] is how much step i wants memory position j. Scaling by sqrt(dim) keeps
        # the dot products from growing with the width and saturating the softmax.
        scores = queries @ memory.transpose(-2, -1) / self.scale  # [batch_size, seq_len, 2 * seq_len]
        # Forbidden positions are pushed to the dtype's minimum rather than deleted, which sends
        # their exponential to zero while leaving every row the same shape.
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len, 2 * seq_len], rows sum to 1

        # Each step's context is its weighted average over the memory. Dropout hits the weights,
        # so a step is occasionally denied one of the positions it wanted to read.
        return self.dropout(weights) @ memory  # [batch_size, seq_len, dim]
