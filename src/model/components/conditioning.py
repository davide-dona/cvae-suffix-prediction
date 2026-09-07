from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Modulation:
    """What the latent does to one sublayer: it scales and shifts the input the sublayer reads,
    and scales the branch the sublayer contributes back to the residual stream.

    Each field is `[batch_size, 1, d_model]`, the middle axis broadcasting over positions: z says
    one thing about the whole suffix, so every position of a sequence is modulated alike. A
    modulation standing for no latent at all carries plain floats instead, which broadcast
    against any shape.
    """

    scale: torch.Tensor | float
    shift: torch.Tensor | float
    gate: torch.Tensor | float


@dataclass(frozen=True)
class LayerModulation:
    """What the latent does to one decoder layer, a `Modulation` per sublayer."""

    self_attention: Modulation
    cross_attention: Modulation
    feedforward: Modulation


# What a decoder with no latent applies: scale 1, shift 0, gate 1, which is the plain pre-norm
# block the modulation is layered onto. Kept as one constant rather than a branch in every
# sublayer, so conditioned and unconditioned decoding are one code path.
_UNCHANGED = Modulation(scale=1.0, shift=0.0, gate=1.0)
UNCONDITIONED = LayerModulation(
    self_attention=_UNCHANGED, cross_attention=_UNCHANGED, feedforward=_UNCHANGED
)


def modulate(hidden: torch.Tensor, modulation: Modulation) -> torch.Tensor:
    """Scale and shift a sublayer's input by what the latent says.

    Args:
        hidden: The normalized input, `[batch_size, seq_len, d_model]`.
        modulation: What to apply.
    Returns:
        `hidden`, the shape it came in at.
    """
    return hidden * modulation.scale + modulation.shift


def gate(branch: torch.Tensor, modulation: Modulation) -> torch.Tensor:
    """Scale what a sublayer contributes back to the residual stream.

    Args:
        branch: The sublayer's output, `[batch_size, seq_len, d_model]`.
        modulation: What to apply.
    Returns:
        `branch`, the shape it came in at.
    """
    return branch * modulation.gate


class AdaLNConditioning(nn.Module):
    """z as an adaptive layer norm: it scales and shifts every sublayer's input and gates every
    sublayer's output, in every layer.

    Unlike a cross-attention token, this path cannot be attenuated by an attention weight: a
    position's representation passes through the modulation whatever the decoder has learned to
    attend to, which is what keeps z carrying information rather than collapsing onto the prior.

    Each layer reads nine `[batch_size, d_model]` vectors off its own head: a scale, a shift and a
    gate for each of the three sublayers. The head is zero-initialized and the scale and gate are
    read as `1 + w(z)`, so a layer starts as exactly the unconditioned pre-norm block it would
    otherwise be, and the modulation is learned from there rather than disturbing the stack at
    step zero.
    """

    # Per sublayer: a scale, a shift and a gate; per layer: three sublayers.
    _PARAMETERS_PER_SUBLAYER = 3
    _SUBLAYERS = 3

    def __init__(self, *, latent_dim: int, d_model: int, num_layers: int):
        super().__init__()
        self.d_model = d_model
        # A nonlinear map of z, shared by every layer: read straight off `latent_dim` instead, the
        # modulation each layer applies could only ever be a linear function of the latent.
        self.trunk = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=d_model), nn.SiLU()
        )
        width = self._PARAMETERS_PER_SUBLAYER * self._SUBLAYERS * d_model
        self.heads = nn.ModuleList(
            nn.Linear(in_features=d_model, out_features=width) for _ in range(num_layers)
        )
        # Zeroed weight and bias alike, so every head emits zeros and every layer starts at
        # scale 1, shift 0, gate 1. The trunk's output is not zero, so the heads still take
        # gradient from the first step.
        for head in self.heads:
            nn.init.zeros_(tensor=head.weight)
            nn.init.zeros_(tensor=head.bias)

    def layers(self, z: torch.Tensor) -> tuple[LayerModulation, ...]:
        """Read every layer's modulation off the latent.

        Read once per pass rather than once per decode step: z is fixed for the whole of a
        generation, so a step costs nothing beyond applying what this returned.

        Args:
            z: The sampled latent, `[batch_size, latent_dim]`.
        Returns:
            One `LayerModulation` per decoder layer, in stack order.
        """
        conditioning = self.trunk(z)  # [batch_size, d_model]
        return tuple(self._modulations(head(conditioning)) for head in self.heads)

    def _modulations(self, parameters: torch.Tensor) -> LayerModulation:
        """Split one head's output into a `Modulation` per sublayer.

        Args:
            parameters: One layer's head output, `[batch_size, 9 * d_model]`.
        Returns:
            The layer's modulation, in sublayer order: self-attention, cross-attention,
            feed-forward.
        """
        # [batch_size, 9, d_model] -> a [batch_size, 1, d_model] slice per parameter, the middle
        # axis left in place so each broadcasts over the positions it modulates.
        split = parameters.unflatten(
            dim=-1, sizes=(self._PARAMETERS_PER_SUBLAYER * self._SUBLAYERS, self.d_model)
        ).unsqueeze(dim=2)  # [batch_size, 9, 1, d_model]
        return LayerModulation(
            *(
                Modulation(
                    scale=1.0 + split[:, index],
                    shift=split[:, index + 1],
                    gate=1.0 + split[:, index + 2],
                )
                for index in range(0, self._PARAMETERS_PER_SUBLAYER * self._SUBLAYERS, 3)
            )
        )
