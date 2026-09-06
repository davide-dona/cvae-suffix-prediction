from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class DecodedEvents:
    """Set of predicted events, in the log's own units. Any new addition
    to the model's output must be added here"""

    # One character per activity, on the dataset's own scale: the codebook seeded from
    # `codec.activity.names`, whose vocabulary the generations file carries in its metadata. Held
    # as a string rather than a list of names so a suffix is one object to compare, to hash and to
    # store, which is what `src/suffixes.py` measures edit distances over.
    activities: str
    # The minutes of cycle time before each activity, in the same order, so a run's timestamps are
    # these accumulated from the last prefix event on.
    cycle_time_minutes: list[float]
    # Minutes until the case ends. Predicted on its own rather than summed from the times above.
    remaining_time_minutes: float

    def __len__(self) -> int:
        return len(self.activities)


@dataclass(frozen=True)
class Draws:
    """One prefix's drawn suffixes, held as the distinct ones and which draw took each.

    The decoder is deterministic given `z`, so two draws that landed on the same activities are one
    sequence the model produced twice: it is written once and `taken` says how often. The cycle
    times do not collapse with it, since those two draws came from different `z` and the decoder
    wrote each its own times, so `events` stays one entry per draw and pairs with
    `suffixes[taken[draw]]`.

    Keeping the draws folded is what lets conformance and the transport cost be solved over the
    distinct suffixes rather than over every draw, which on a collapsed run is most of the work.
    """

    # The distinct suffixes, in the order they were first drawn
    suffixes: tuple[str, ...]
    # Which of them each draw took, one entry per draw in the order they were drawn:
    # `hit_rate_at_k` reads the first k.
    taken: tuple[int, ...]
    # The cycle times and the remaining time of each draw, in the same order as `taken`. The
    # activities of draw `i` are `suffixes[taken[i]]`.
    events: list[DecodedEvents]

    @classmethod
    def of(cls, drawn: list[DecodedEvents]) -> 'Draws':
        """Fold one prefix's draws, in the order they were drawn.

        Args:
            drawn: One entry per draw of `z`, already decoded.
        Returns:
            The draws with their distinct suffixes pulled out.
        """
        rows: dict[str, int] = {}
        taken = tuple(rows.setdefault(events.activities, len(rows)) for events in drawn)
        return cls(suffixes=tuple(rows), taken=taken, events=drawn)

    def __len__(self) -> int:
        """How many draws were taken, which is what every mean over them divides by."""
        return len(self.taken)

    @cached_property
    def counts(self) -> np.ndarray:
        """How many draws landed on each of `suffixes`, in the same order.

        The weights every score over the distinct suffixes reads: a mean over the draws is the
        weighted mean over these, and they sum to `len(self)`.
        """
        return np.bincount(self.taken, minlength=len(self.suffixes)).astype(np.float64)


@dataclass(frozen=True)
class Generation:
    """One prefix's generated suffixes, the point prediction beside them, and the truth they were
    generated for.
    - case_id is used to identify the case in the log the prefix was cut from;
    - prefix_activities are added for convenience in the conformance report.
    """

    case_id: str  # which case of the log the prefix was cut from
    prefix_activities: str  # the events before the cut, in order, one character each
    samples: Draws  # every draw of z, folded onto the distinct suffixes they took
    point: DecodedEvents  # the suffix written from the mean of `p(z | prefix)`
    truth: DecodedEvents

    @property
    def prefix_len(self) -> int:
        """The cut point this answers: how many events ran before it."""
        return len(self.prefix_activities)
