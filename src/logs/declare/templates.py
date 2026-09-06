from collections.abc import Callable
from dataclasses import dataclass

# The positions of each activity in a trace, keyed by the activity's character.
Positions = dict[str, list[int]]


@dataclass(frozen=True, slots=True)
class Constraint:
    """One constraint of a model: the template it follows and the activities it is about."""

    template: '_Template'
    # The first activity named, which is what activates every template but the precedence family
    first: str
    # The second activity of a binary template, or None for a unary one
    second: str | None
    # How many occurrences of `first` a counting template asks for, and 1 for the rest
    n: int

    def holds(self, trace: str, positions: Positions) -> bool:
        """Whether one finished trace satisfies this constraint.
        Args:
            trace: The trace's activities, one character each, in order.
            positions: Where each of them occurs, from `ConformanceChecker.check`.
        Returns:
            True if the trace both activates the constraint and does not violate it.
        """
        return self.template.holds(self, trace, positions)


@dataclass(frozen=True, slots=True)
class _Template:
    """One DECLARE template: what satisfying it means, and how a constraint of it is written."""

    holds: Callable[[Constraint, str, Positions], bool]
    is_binary: bool
    supports_cardinality: bool


def _existence(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` occurs at least `n` times."""
    return len(positions.get(constraint.first, ())) >= constraint.n


def _absence(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` occurs fewer than `n` times, never running it included."""
    return len(positions.get(constraint.first, ())) < constraint.n


def _exactly(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` occurs exactly `n` times."""
    return len(positions.get(constraint.first, ())) == constraint.n


def _init(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` is the first event of the trace."""
    return bool(trace) and trace[0] == constraint.first


def _end(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` is the last event of the trace."""
    return bool(trace) and trace[-1] == constraint.first


def _choice(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` or `b` occurs."""
    return constraint.first in positions or constraint.second in positions


def _exclusive_choice(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` or `b` occurs, and never both."""
    return (constraint.first in positions) != (constraint.second in positions)


def _responded_existence(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` occurs, and so does `b`."""
    return constraint.first in positions and constraint.second in positions


def _not_responded_existence(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` occurs, and `b` does not."""
    return constraint.first in positions and constraint.second not in positions


def _response(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` occurs, and every occurrence of it is followed by a `b`."""
    activations = positions.get(constraint.first)
    targets = positions.get(constraint.second)
    return bool(activations) and bool(targets) and activations[-1] < targets[-1]


def _precedence(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`b` occurs, and every occurrence of it is preceded by an `a`."""
    activations = positions.get(constraint.second)
    earlier = positions.get(constraint.first)
    return bool(activations) and bool(earlier) and earlier[0] < activations[0]


def _not_response(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` occurs, and no occurrence of it is followed by a `b`."""
    activations = positions.get(constraint.first)
    targets = positions.get(constraint.second)
    return bool(activations) and (not targets or activations[0] > targets[-1])


def _not_precedence(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`b` occurs, and no occurrence of it is preceded by an `a`."""
    activations = positions.get(constraint.second)
    earlier = positions.get(constraint.first)
    return bool(activations) and (not earlier or earlier[0] > activations[-1])


def _chain_response(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` occurs, and a `b` follows it immediately every time."""
    activations = positions.get(constraint.first)
    last = len(trace) - 1
    return bool(activations) and all(
        index < last and trace[index + 1] == constraint.second for index in activations
    )


def _chain_precedence(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`b` occurs, and an `a` precedes it immediately every time."""
    activations = positions.get(constraint.second)
    return bool(activations) and all(
        index > 0 and trace[index - 1] == constraint.first for index in activations
    )


def _not_chain_response(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` occurs, and a `b` never follows it immediately."""
    activations = positions.get(constraint.first)
    last = len(trace) - 1
    return bool(activations) and not any(
        index < last and trace[index + 1] == constraint.second for index in activations
    )


def _not_chain_precedence(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`b` occurs, and an `a` never precedes it immediately."""
    activations = positions.get(constraint.second)
    return bool(activations) and not any(
        index > 0 and trace[index - 1] == constraint.first for index in activations
    )


def _alternate_response(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`a` occurs, and a `b` follows each occurrence of it before `a` recurs."""
    activations = fulfillments = 0
    pending = False
    for activity in trace:
        if activity == constraint.first:
            pending = True
            activations += 1
        if pending and activity == constraint.second:
            pending = False
            fulfillments += 1
    return activations > 0 and activations == fulfillments


def _alternate_precedence(constraint: Constraint, trace: str, positions: Positions) -> bool:
    """`b` occurs, and an `a` precedes each occurrence of it since the previous `b`."""
    activations = fulfillments = 0
    preceding = 0
    for activity in trace:
        if activity == constraint.first:
            preceding += 1
        if activity == constraint.second:
            activations += 1
            if preceding:
                fulfillments += 1
            preceding = 0
    return activations > 0 and activations == fulfillments


# Every template `pipelines.preprocess` can mine, keyed by the name it writes into the model
# file.
TEMPLATES: dict[str, _Template] = {
    'Existence': _Template(holds=_existence, is_binary=False, supports_cardinality=True),
    'Absence': _Template(holds=_absence, is_binary=False, supports_cardinality=True),
    'Exactly': _Template(holds=_exactly, is_binary=False, supports_cardinality=True),
    'Init': _Template(holds=_init, is_binary=False, supports_cardinality=False),
    'End': _Template(holds=_end, is_binary=False, supports_cardinality=False),
    'Choice': _Template(holds=_choice, is_binary=True, supports_cardinality=False),
    'Exclusive Choice': _Template(
        holds=_exclusive_choice, is_binary=True, supports_cardinality=False
    ),
    'Responded Existence': _Template(
        holds=_responded_existence, is_binary=True, supports_cardinality=False
    ),
    'Not Responded Existence': _Template(
        holds=_not_responded_existence, is_binary=True, supports_cardinality=False
    ),
    'Response': _Template(holds=_response, is_binary=True, supports_cardinality=False),
    'Precedence': _Template(holds=_precedence, is_binary=True, supports_cardinality=False),
    'Not Response': _Template(holds=_not_response, is_binary=True, supports_cardinality=False),
    'Not Precedence': _Template(holds=_not_precedence, is_binary=True, supports_cardinality=False),
    'Chain Response': _Template(holds=_chain_response, is_binary=True, supports_cardinality=False),
    'Chain Precedence': _Template(
        holds=_chain_precedence, is_binary=True, supports_cardinality=False
    ),
    'Not Chain Response': _Template(
        holds=_not_chain_response, is_binary=True, supports_cardinality=False
    ),
    'Not Chain Precedence': _Template(
        holds=_not_chain_precedence, is_binary=True, supports_cardinality=False
    ),
    'Alternate Response': _Template(
        holds=_alternate_response, is_binary=True, supports_cardinality=False
    ),
    'Alternate Precedence': _Template(
        holds=_alternate_precedence, is_binary=True, supports_cardinality=False
    ),
}
