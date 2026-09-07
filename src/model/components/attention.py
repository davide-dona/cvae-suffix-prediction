from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ProjectedKeysValues:
    """One attention input, already projected and split across heads.

    A value of its own because projecting is the work worth not repeating. The encoded prefix
    does not change while a suffix is being written, and a suffix position already written does
    not change when the next one is added, so both can be projected once and kept.
    """

    keys: torch.Tensor  # [batch_size, num_heads, source_len, head_dim]
    values: torch.Tensor  # [batch_size, num_heads, source_len, head_dim]


class MultiHeadAttention(nn.Module):
    """Multi-head attention with cached key/value projections.

    `nn.MultiheadAttention` re-projects the entire source at every decoding step; here `project`
    runs once and its result is reused for each step.
    """

    def __init__(self, *, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.query_projection = nn.Linear(in_features=d_model, out_features=d_model)
        self.key_projection = nn.Linear(in_features=d_model, out_features=d_model)
        self.value_projection = nn.Linear(in_features=d_model, out_features=d_model)
        self.output_projection = nn.Linear(in_features=d_model, out_features=d_model)

        # Matches `nn.MultiheadAttention`: Xavier uniform q/k/v weights, zero biases throughout,
        # output weights at their default.
        for projection in (self.query_projection, self.key_projection, self.value_projection):
            nn.init.xavier_uniform_(tensor=projection.weight)
            nn.init.zeros_(tensor=projection.bias)
        nn.init.zeros_(tensor=self.output_projection.bias)

    def project(self, source: torch.Tensor) -> ProjectedKeysValues:
        """Turn a sequence into the keys and values attention reads it as.
        Args:
            source: `[batch_size, source_len, d_model]`.
        Returns:
            Its keys and values, split across heads.
        """
        return ProjectedKeysValues(
            keys=self._split_heads(self.key_projection(source)),
            values=self._split_heads(self.value_projection(source)),
        )

    def forward(
        self,
        query: torch.Tensor,
        keys_values: ProjectedKeysValues,
        *,
        key_padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Attend from `query` over `keys_values`.
        Args:
            query: `[batch_size, query_len, d_model]`.
            keys_values: What to attend over, from `project`.
            key_padding_mask: True where a source position holds padding,
                `[batch_size, source_len]`.
            causal: Whether a query position may only attend over itself and what precedes it.
                Valid only when the query covers every key, which is the un-cached pass.
        Returns:
            `[batch_size, query_len, d_model]`.
        """
        batch_size, query_len, d_model = query.shape
        queries = self._split_heads(
            self.query_projection(query)
        )  # [batch_size, num_heads, query_len, head_dim]

        attention_mask = None
        if key_padding_mask is not None:
            attention_mask = ~key_padding_mask[:, None, None, :]  # [batch_size, 1, 1, source_len]

        attended = F.scaled_dot_product_attention(
            query=queries,
            key=keys_values.keys,
            value=keys_values.values,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )  # [batch_size, num_heads, query_len, head_dim]

        merged = attended.transpose(dim0=1, dim1=2).reshape(batch_size, query_len, d_model)
        return self.output_projection(merged)

    def _split_heads(self, projected: torch.Tensor) -> torch.Tensor:
        """`[batch_size, length, d_model]` -> `[batch_size, num_heads, length, head_dim]`,
        giving each head its own slice of the embedding dimension."""
        batch_size, length, _ = projected.shape
        return projected.view(batch_size, length, self.num_heads, self.head_dim).transpose(
            dim0=1, dim1=2
        )
