from collections.abc import Sequence
from dataclasses import asdict, fields
from typing import Self

from torch.utils.tensorboard import SummaryWriter


def mean(values: Sequence[float]) -> float:
    """The mean of `values`, or 0.0 if there are none."""
    return sum(values) / len(values) if values else 0.0


class ScalarMetrics:
    """Named scalars of one pass, aggregated field by field.

    Two aggregations, because a pass comes in two shapes. A loss is accumulated with `+` as the
    batches arrive and divided once by the traces it summed over, so a trace weighs the same
    however the batches fell. A per-prefix score is already reduced when it is produced, so it is
    averaged with `mean` over the prefixes themselves.
    """

    @classmethod
    def mean(cls, values: Sequence[Self]) -> Self:
        """Average a set of metrics field by field.

        Args:
            values: The metrics to average, one per unit of the pass.
        Returns:
            The mean of every field, all 0.0 if there are none.
        """
        return cls(
            **{
                field.name: (
                    sum(getattr(value, field.name) for value in values) / len(values)
                    if values
                    else 0.0
                )
                for field in fields(cls)
            }
        )

    def __add__(self, other: Self) -> Self:
        return type(self)(
            **{
                field.name: getattr(self, field.name) + getattr(other, field.name)
                for field in fields(self)
            }
        )

    def __truediv__(self, divisor: float) -> Self:
        return type(self)(
            **{field.name: getattr(self, field.name) / divisor for field in fields(self)}
        )

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
