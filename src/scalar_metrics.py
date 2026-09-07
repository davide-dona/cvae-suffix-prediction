from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from typing import Any, Self

import numpy as np
import wandb


def mean(values: Sequence[float]) -> float:
    """The mean of `values`, or 0.0 if there are none."""
    return sum(values) / len(values) if values else 0.0


class Unit(StrEnum):
    """What a metric's number is: what is written after its name, and the range it is read over.

    The two are one question. A share is a share whether it is drawn or tabulated, and it is
    exactly what carries no unit and spans `[0, 1]`; a count of days is what carries `days` and
    starts at 0 with no ceiling.
    """

    SHARE = 'share'  # A fraction of something, in [0, 1]
    SCORE = 'score'  # Dimensionless with no fixed range, e.g. a distance between two sets
    COUNT = 'count'  # How many of something there were, starting at 0 with no ceiling
    DAYS = 'days'
    EVENTS = 'events'

    @property
    def symbol(self) -> str:
        """What is written after the metric's name, or `''` for a dimensionless number."""
        return '' if self in (Unit.SHARE, Unit.SCORE, Unit.COUNT) else str(self)

    @property
    def bounds(self) -> tuple[float | None, float | None]:
        """The range a value of this unit is read over, either end `None` where the data sets it.

        A share spans the whole of `[0, 1]`, so a panel of one shows where a model sits within
        what the metric can be rather than within the range these models happen to cover. An
        absolute error and a count both start at 0 and have no ceiling. A score is bounded at
        neither end.
        """
        if self is Unit.SHARE:
            return 0.0, 1.0
        if self is Unit.SCORE:
            return None, None
        return 0.0, None


class Direction(StrEnum):
    """Which way one of a report's numbers reads: the arrow written after its name, and what the
    emphasis of a table calls the best of a row.

    A gap has a target rather than a direction, its two signs being two ways of being wrong, so it
    is read on its distance from 0 instead. A property of the log is not a model's score at all,
    and neither is a spread whose target is the log's own, so neither is ever emphasized.
    """

    HIGHER = 'higher'
    LOWER = 'lower'
    ZERO = 'zero'  # A gap, best at 0, either sign being a way of being wrong
    NONE = 'none'  # No better value, e.g. a spread or a property of the log

    @property
    def sign(self) -> float:
        """Which way a value has to be turned for more to be better: `+1.0` or `-1.0`.

        A gap is negative too, being read on its distance from 0 once `reads_absolute` has taken
        that distance. A metric with no direction has no better value and is never turned, so it is
        `+1.0` by convention rather than by meaning.
        """
        return -1.0 if self in (Direction.LOWER, Direction.ZERO) else 1.0

    @property
    def reads_absolute(self) -> bool:
        """Whether the value is read on its distance from 0 rather than on its own sign."""
        return self is Direction.ZERO


class Owner(StrEnum):
    """Whose number one of a report's metrics is: the model that was run, or the log it was run
    against.

    Every model of a log carries the same value for a log's metric, so the two are drawn and
    written differently. A log's own value is one series in its own style rather than one per
    model in each model's colour, and it is never a row of a table comparing models, being the
    target they are read against rather than a competitor.
    """

    MODEL = 'model'
    LOG = 'log'


def oriented(values: np.ndarray, directions: Sequence[Direction]) -> np.ndarray:
    """Rewrite a set of values so that more is better on every metric.

    The one place a `Direction` becomes arithmetic, so a comparison reads the same way whichever
    way the metric does.

    Args:
        values: Values whose last axis is the metrics, of any rank above that: `[..., metrics]`.
        directions: Which way each of those metrics reads, in that same order.
    Returns:
        The same shape, each metric negated where lower is better and taken absolute first where it
        is a gap. Applied to a mean rather than to one prefix's own score: the number a table
        prints is the mean gap, so its distance from 0 is what a test reads.
    """
    # [metrics] each, so a call at any rank broadcasts over the leading axes.
    sign = np.array([direction.sign for direction in directions], dtype=np.float64)
    absolute = np.array([direction.reads_absolute for direction in directions], dtype=bool)
    return sign * np.where(absolute, np.abs(values), values)


@dataclass(frozen=True)
class Metric:
    """What one of the numbers a report carries is: the name it goes by, its unit, and which way
    it reads.

    What it is called on a page is not here: a table holding one estimator throughout names a
    metric more shortly than a figure drawing that estimator against another, so the name belongs
    to the appearance rather than to the metric.
    """

    key: str  # The field it was declared on, which is the name it has in a report and a frame
    unit: Unit
    direction: Direction = Direction.NONE
    owner: Owner = Owner.MODEL


# Where a field's declaration is kept. Read through `metrics_of` rather than directly.
_DECLARATION = 'metric'


def metric(*, unit: Unit, direction: Direction = Direction.NONE, owner: Owner = Owner.MODEL) -> Any:
    """Declare a float field as one of the numbers a report carries.

    Args:
        unit: What the number is, which gives it both its printed unit and its range.
        direction: Which way it reads, which is both the arrow written after its name and what a
            table's emphasis calls the best of a row.
        owner: Whose number it is, which is the series it is drawn as and whether it is ever
            written as a model's row.
    Returns:
        A `dataclasses.field` carrying the declaration. The metric's key is the field's own name,
        so it is never written twice and cannot drift from what a report holds.
    """
    return field(metadata={_DECLARATION: (unit, direction, owner)})


def metrics_of(cls: type) -> tuple[Metric, ...]:
    """Every field of a dataclass declared with `metric`, in declaration order.

    Args:
        cls: The dataclass to read. A field declared without `metric` is not one of the numbers a
            report carries and is skipped, which is what lets a class hold both.
    Returns:
        One `Metric` per declared field, keyed by the field's name.
    """
    declared = []
    for entry in fields(cls):
        if _DECLARATION not in entry.metadata:
            continue
        unit, direction, owner = entry.metadata[_DECLARATION]
        declared.append(Metric(key=entry.name, unit=unit, direction=direction, owner=owner))
    return tuple(declared)


class ScalarMetrics:
    """Named scalars of one pass, aggregated field by field.

    Two aggregations, because a pass comes in two shapes. A loss is accumulated with `+` as the
    batches arrive and divided once by the traces it summed over, so a trace weighs the same
    however the batches fell. A per-prefix score is already reduced when it is produced, so it is
    averaged with `mean` over the prefixes themselves.
    """

    # Subclasses declare `slots=True`, which only saves anything if this carries no `__dict__`.
    __slots__ = ()

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
                field.name: mean([getattr(value, field.name) for value in values])
                for field in fields(cls)
            }
        )

    @classmethod
    def metrics(cls) -> tuple[Metric, ...]:
        """Every field of a family of scores, each being one of the numbers a report carries.

        Asked only of a family a report is written from. `Loss` is logged rather than reported, so
        it declares nothing and never calls this.

        Returns:
            One `Metric` per field, in declaration order.
        Raises:
            ValueError: If a field was not declared with `metric`, naming it. A score added to a
                family and left undeclared would reach every report and no figure.
        """
        declared = metrics_of(cls)
        keys = {declaration.key for declaration in declared}
        undeclared = [entry.name for entry in fields(cls) if entry.name not in keys]
        if undeclared:
            raise ValueError(
                f'{cls.__name__} declares no unit or direction for {", ".join(undeclared)}. '
                f'Give each of them a `metric(unit=...)` default, since every field of a family of '
                f'scores is one of the numbers a report carries.'
            )
        return declared

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

    def log(self, step: int, *, prefix: str) -> None:
        """
        Log every field to the active W&B run under a shared prefix.

        Args:
            step: The step these metrics belong to.
            prefix: Namespace to log under, e.g. `train` or `val`, so the two passes of a
                validation line up on the same chart.
        """
        wandb.log({f'{prefix}/{name}': value for name, value in asdict(self).items()}, step=step)
