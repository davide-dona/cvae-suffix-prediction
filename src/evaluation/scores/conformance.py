from dataclasses import dataclass
from typing import Self

from src.inference.generation import Generation
from src.logs.declare import ConformanceChecker
from src.scalar_metrics import Direction, Owner, ScalarMetrics, Unit, metric


@dataclass(frozen=True, slots=True)
class ConformanceScores(ScalarMetrics):
    """Conformance of generated suffixes to process constraints."""

    # Mean constraint-satisfaction share over draws.
    conformance_mean: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    # Constraint-satisfaction share of the point prediction.
    conformance_point: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    # Constraint-satisfaction share of the observed suffix.
    conformance_truth: float = metric(unit=Unit.SHARE, owner=Owner.LOG)

    # Fraction of draws satisfying every constraint.
    full_conformance_mean: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    # Whether the point prediction satisfies every constraint.
    full_conformance_point: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    # Whether the observed suffix satisfies every constraint.
    full_conformance_truth: float = metric(unit=Unit.SHARE, owner=Owner.LOG)

    @classmethod
    def of(cls, generation: Generation, *, checker: ConformanceChecker) -> Self:
        """Score generated, point, and observed suffixes against constraints.

        Args:
            generation: Decoded model output for one prefix.
            checker: Process-constraint checker.

        Returns:
            Conformance scores for the prefix.
        """
        # Constraints apply to the complete trace.
        prefix = generation.prefix_activities
        samples = generation.samples

        # Check each distinct suffix once, then weight by draw count.
        checks = [checker.check(prefix + suffix) for suffix in samples.suffixes]
        draws = len(samples)

        def over_draws(values: list[float]) -> float:
            """The mean over the draws of a value read once per distinct suffix."""
            return float(samples.counts @ values) / draws if values and draws else 0.0

        point = checker.check(prefix + generation.point.activities)
        truth = checker.check(prefix + generation.truth.activities)

        return cls(
            conformance_mean=over_draws([check.share for check in checks]),
            conformance_point=point.share,
            conformance_truth=truth.share,
            full_conformance_mean=over_draws([check.full for check in checks]),
            full_conformance_point=point.full,
            full_conformance_truth=truth.full,
        )
