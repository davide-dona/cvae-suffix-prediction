from dataclasses import dataclass, replace
from functools import lru_cache

from src import paths
from src.logs.declare.constraints import read_constraints
from src.logs.declare.templates import Positions
from src.suffixes import ActivityCodes

# Stands in for an activity the dataset's codebook does not know, so its constraint can never be
# activated by a trace the codebook spelled.
_UNMATCHABLE = '\x00'


@dataclass(frozen=True, slots=True)
class Conformance:
    """How one trace fared against a declarative model, read either as a share or as a verdict.

    The two are the same check at two granularities. A share says how much of the process a trace
    respects, which moves smoothly and so separates models that are all wrong in different amounts;
    the verdict says whether the trace is one the process allows at all, which is what a trace
    handed to someone as a continuation has to be. A trace can sit high on the first and fail the
    second on one constraint.
    """

    satisfied: int
    total: int

    @property
    def share(self) -> float:
        """The fraction of constraints the trace satisfies, in `[0, 1]`, or 0.0 for a model that
        checks nothing."""
        return self.satisfied / self.total if self.total else 0.0

    @property
    def full(self) -> float:
        """1.0 if the trace satisfies every constraint and 0.0 otherwise, so that a mean over
        traces is the share of them that are conformant. A model that checks nothing rates 0.0
        here as it does on `share`, rather than calling every trace conformant."""
        return float(self.total > 0 and self.satisfied == self.total)


class ConformanceChecker:
    """Scores traces against the declarative model a dataset was mined for."""

    def __init__(self, dataset: str, codes: ActivityCodes) -> None:
        """
        Args:
            dataset: The dataset whose model to check against, read from where preprocessing
                wrote it.
            codes: The dataset's codebook, which the constraints are translated onto so a trace is
                checked as the string the generations already hold it as, with nothing decoded per
                check. An activity the codebook does not know is given a character no trace can
                contain, leaving its constraint unactivated rather than growing the codebook.
        """
        self._constraints = tuple(
            replace(
                constraint,
                first=codes.codes.get(constraint.first, _UNMATCHABLE),
                second=(
                    None
                    if constraint.second is None
                    else codes.codes.get(constraint.second, _UNMATCHABLE)
                ),
            )
            for constraint in read_constraints(paths.DECLARE_MODEL.require(dataset))
        )

    @lru_cache(maxsize=100_000)  # noqa: B019 -- one checker per scoring process
    def check(self, trace: str) -> Conformance:
        """
        Check one trace against every constraint of the model.

        Args:
            trace: The trace's activities, one character each, in order, on the dataset's own
                scale. A whole case, prefix included: a constraint like `Init` or `Precedence` is
                about the trace, not about a run of events inside it.
        Returns:
            How many constraints the trace satisfies out of how many there are, which both the
            share and the verdict are read off.
        """
        positions: Positions = {}
        for index, activity in enumerate(trace):
            positions.setdefault(activity, []).append(index)

        satisfied = sum(constraint.holds(trace, positions) for constraint in self._constraints)
        return Conformance(satisfied=satisfied, total=len(self._constraints))
