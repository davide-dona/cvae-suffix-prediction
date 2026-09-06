from dataclasses import dataclass

from src.scalar_metrics import ScalarMetrics


@dataclass(frozen=True)
class Loss(ScalarMetrics):
    """The loss of one pass, and the terms it is made of.

    One shape for both architectures, so a curve reads the same whichever produced it - each
    computes it in `SuffixModel.compute_loss`, and a model with no latent leaves
    `floored_kl_loss` at 0.0 and its `loss` is the reconstruction alone.
    """

    loss: float = 0.0
    reconstruction_loss: float = 0.0
    # The KL after each dimension is floored at `free_bits`, which is what `loss` charges for.
    # The weight it is charged at is `kl_weight`, logged on its own.
    floored_kl_loss: float = 0.0
    activity_loss: float = 0.0
    cycle_time_loss: float = 0.0
    remaining_time_loss: float = 0.0
    # What each time head was charged for the scale it emitted, already inside the two terms above
    # rather than added to them. Subtracting one leaves the absolute error every architecture pays,
    # so a time curve reads the same whichever arm produced it; a model whose time heads carry no
    # scale leaves both at 0.0 and its two terms are that error outright.
    cycle_time_scale_loss: float = 0.0
    remaining_time_scale_loss: float = 0.0
