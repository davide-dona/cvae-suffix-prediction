from dataclasses import dataclass
from typing import Self

from src.inference.generation import Generation
from src.logs.declare import ConformanceChecker
from src.scalar_metrics import Direction, Owner, ScalarMetrics, Unit, metric


@dataclass(frozen=True, slots=True)
class ConformanceScores(ScalarMetrics):
    """Whether a prefix's generated suffixes are traces the process allows at all.

    The scores of one prefix's generated samples and of the suffix it actually took, or their mean
    over a set of prefixes. Each level is read twice: as the share of the model's constraints a
    trace satisfies, and as whether it satisfies all of them. The share says how far off a trace is
    where the verdict says whether it is a trace the process admits, and a set of traces can sit
    high on the first while almost none of them passes the second.
    """

    # The mean over a prefix's samples
    conformance_mean: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    # The suffix written from the mean of `p(z | prefix)`
    conformance_point: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    # The suffix the log actually took, through the same checker. A property of the log rather
    # than of the model, so it is the same for every model of a log and is never compared between
    # them: it is the target the two levels above are read against.
    conformance_truth: float = metric(unit=Unit.SHARE, owner=Owner.LOG)

    # The share of a prefix's draws that satisfy every constraint, so a mean over prefixes is the
    # share of the generated traces that are fully conformant.
    full_conformance_mean: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    # Whether the point prediction satisfies every constraint, 1.0 or 0.0
    full_conformance_point: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    # Whether the suffix the log took satisfies every constraint. The log's own, as above: a
    # discovered model does not hold on every case it was mined from, so this is well under 1.0
    # and is the ceiling the two levels above are read against rather than a perfect score.
    full_conformance_truth: float = metric(unit=Unit.SHARE, owner=Owner.LOG)

    @classmethod
    def of(cls, generation: Generation, *, checker: ConformanceChecker) -> Self:
        """Check one prefix's generated suffixes, its point prediction and its ground truth.

        Args:
            generation: The model's answer for one prefix, decoded into the log's own units.
            checker: The declarative model to check against, held by the process doing the scoring.
        Returns:
            The prefix's conformance and that of the suffix the log took, which is the target the
            two levels are read against. A prefix with no samples scores 0.0 on the two sampled
            metrics, the worst they can be, rather than looking perfectly conformant.
        """
        # A constraint is about the whole trace, so a suffix is checked as the case it completes.
        prefix = generation.prefix_activities
        samples = generation.samples

        # One check per distinct suffix, weighed by how many draws took it. A repeated draw is the
        # same trace and rates the same, so these are the means over the draws with the constraints
        # walked once per distinct suffix rather than once per draw.
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
