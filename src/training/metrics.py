from dataclasses import asdict, fields
from typing import Self

from torch.utils.tensorboard import SummaryWriter


class ScalarMetrics:
    """Named scalars of one pass, summed over its traces and divided once at the end."""

    def __add__(self, other: Self) -> Self:
        return type(self)(**{
            field.name: getattr(self, field.name) + getattr(other, field.name)
            for field in fields(self)
        })

    def __truediv__(self, divisor: float) -> Self:
        return type(self)(**{
            field.name: getattr(self, field.name) / divisor for field in fields(self)
        })

    def log(self, writer: SummaryWriter, step: int, *, prefix: str) -> None:
        """
        Write every field to TensorBoard under a shared prefix.

        Args:
            writer: The TensorBoard writer to log to.
            step: The step these metrics belong to.
            prefix: Namespace to log under, e.g. `train` or `val`, so the two passes of a
                validation line up on the same chart.
        """
        for name, value in asdict(self).items():
            writer.add_scalar(f'{prefix}/{name}', value, step)
