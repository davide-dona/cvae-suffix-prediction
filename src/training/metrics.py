from dataclasses import asdict, dataclass, fields
from typing import Sequence

from torch.utils.tensorboard import SummaryWriter


@dataclass
class Metrics:
    """The loss of one pass, and the terms it is made of.

    The same numbers are reported at two scales: `compute_loss` returns them summed over one
    batch, and `src/training/train.py` divides an accumulation of them by the traces that went
    into it. The arithmetic below is what turns the first into the second, and the fields
    default to zero so that a bare `Metrics()` is the accumulator batches are added into.
    """
    loss: float = 0.0
    reconstruction_loss: float = 0.0
    kl_loss: float = 0.0
    activity_loss: float = 0.0
    time_delta_loss: float = 0.0

    def __add__(self, other: "Metrics") -> "Metrics":
        return Metrics(
            loss=self.loss + other.loss,
            reconstruction_loss=self.reconstruction_loss + other.reconstruction_loss,
            kl_loss=self.kl_loss + other.kl_loss,
            activity_loss=self.activity_loss + other.activity_loss,
            time_delta_loss=self.time_delta_loss + other.time_delta_loss,
        )

    def __truediv__(self, divisor: float) -> "Metrics":
        return Metrics(
            loss=self.loss / divisor,
            reconstruction_loss=self.reconstruction_loss / divisor,
            kl_loss=self.kl_loss / divisor,
            activity_loss=self.activity_loss / divisor,
            time_delta_loss=self.time_delta_loss / divisor,
        )

    def log(self, writer: SummaryWriter, step: int, *, prefix: str) -> None:
        """
        Write every metric to TensorBoard under a shared prefix.
        Args:
            writer: The TensorBoard writer to log to.
            step: The epoch the metrics belong to.
            prefix: Namespace to log under, e.g. `train` or `val`, so the two passes of an
                epoch line up on the same chart.
        """
        for name, value in asdict(self).items():
            writer.add_scalar(f'{prefix}/{name}', value, step)

    @classmethod
    def log_layout(cls, writer: SummaryWriter, *, prefixes: Sequence[str]) -> None:
        """
        Write the layout that also draws a metric's several passes as one chart.

        The prefixed tags `log` writes are one chart each, which is what keeps a metric
        comparable across runs; this overlays them in TensorBoard's Custom Scalars
        dashboard, where `train/loss`, `val/loss` and `val_prior/loss` are drawn together
        instead. It is a second view of the same tags and writes no numbers of its own, so
        nothing is duplicated. Read from the fields, so adding a metric still means adding
        a field.

        Args:
            writer: The TensorBoard writer of the run to lay out. Written once per run,
                since a run has one layout.
            prefixes: The namespaces `log` is called with over the run, in the order they
                should be drawn.
        """
        writer.add_custom_scalars({
            'passes': {
                field.name: ['Multiline', [f'{prefix}/{field.name}' for prefix in prefixes]]
                for field in fields(cls)
            }
        })


@dataclass(frozen=True)
class GenerationMetrics:
    """What the model produced when it was made to write suffixes on its own.

    Everything in `Metrics` is teacher-forced: the decoder is handed the true event at every
    position and only ever has to pick the next one. This is the other regime, the one
    `generate` and therefore `pipelines/evaluate.py` run in, where a mistake at position 3 is
    still there at position 4. It is the only thing in a run that measures what the model is
    for, which is why it is what best-model selection and early stopping read.

    The first two are `src/evaluation/accuracy.py`'s `activity_dls_mean` and
    `activity_dls_best`, computed the same way over a fixed slice of the validation split, so
    a training curve and a final report are the same number.

    `sample_diversity` is the third because a CVAE can fail in either direction, and the two
    failures look nothing alike on a loss curve. If the KL term does too little, the posterior
    smuggles the suffix through z and the prior-path numbers fall apart. If it does too much,
    z carries nothing, all `num_samples` suffixes of a prefix come back identical, and the
    model is a deterministic sequence-to-sequence model wearing a latent variable.
    """
    activity_dls_mean: float = 0.0
    activity_dls_best: float = 0.0
    sample_diversity: float = 0.0

    def log(self, writer: SummaryWriter, step: int, *, prefix: str = 'gen') -> None:
        """Write every metric to TensorBoard under a shared prefix."""
        for name, value in asdict(self).items():
            writer.add_scalar(f'{prefix}/{name}', value, step)
