from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Self

import numpy as np
import ot
from scipy.stats import wasserstein_distance

from src.evaluation.scores.accuracy import MINUTES_PER_DAY
from src.inference.generation import Generation
from src.logs import ContinuationIndex, Continuations
from src.scalar_metrics import Direction, Owner, ScalarMetrics, Unit, metric
from src.suffixes import distances, diversity

# How many times the log has to have run a prefix before its draws are read as a distribution
# at all. Below it the reference is a sample too small to be one: at a single occurrence the
# transport problem has one column and every score here collapses onto an accuracy, `emsc`
# becoming `dls_mean`, `length_wasserstein` becoming `length_ae_mean` and
# `remaining_time_wasserstein_days` becoming `remaining_time_ae_mean_days`, each to machine
# precision, and an expected distance to one suffix is maximized by drawing that suffix every
# time. Five is where the log's own distribution stops losing to a sharpened version of itself.
MIN_REFERENCE_OCCURRENCES = 5


@dataclass(frozen=True, slots=True)
class DistributionScores(ScalarMetrics):
    """How close the distribution of suffixes generated for a prefix is to the distribution of
    the ones the log was observed to take after it, and how far apart each of the two spreads its
    own continuations."""

    # Earth Movers' Stochastic Conformance between the two distributions, as a similarity: the
    # optimal transport cost of turning the generated suffixes into the observed ones, subtracted
    # from 1, so 1.0 means the model reproduces the prefix's continuations exactly.
    emsc: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)

    # The precision and recall of the samples read against the observed continuations. Recall is
    # weighted by how often each continuation was observed, so it is the share of what actually
    # happens that the model reproduces; precision counts a draw as a hit whether the log took that
    # continuation once or five hundred times.
    continuation_recall: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)
    continuation_precision: float = metric(unit=Unit.SHARE, direction=Direction.HIGHER)

    # The comparison on the three marginals a suffix carries beyond its activities. The cycle times
    # are grouped by the activity they precede rather than by the position they fell at: a process
    # constrains how long an activity takes, where a position only means something once the
    # control flow is already right, which the three columns above are what answer.
    length_wasserstein: float = metric(unit=Unit.EVENTS, direction=Direction.LOWER)
    remaining_time_wasserstein_days: float = metric(unit=Unit.DAYS, direction=Direction.LOWER)
    activity_time_wasserstein_days: float = metric(unit=Unit.DAYS, direction=Direction.LOWER)

    # How far apart the samples of a prefix are, and how many of them are distinct sequences.
    # Neither has a best value of its own: the spread a model should have is the spread the log's
    # own continuations have, which `reference_diversity` below carries.
    sample_diversity: float = metric(unit=Unit.SHARE)
    unique_sample_rate: float = metric(unit=Unit.SHARE)

    # How far apart two of the prefix's observed continuations are, which is the scale
    # `sample_diversity` is read against: a model matching the log's multimodality spreads its
    # draws about this far apart, no further and no less. A property of the log rather than of the
    # model, so it is the same for every model of a log and is never compared between them.
    reference_diversity: float = metric(unit=Unit.SHARE, owner=Owner.LOG)

    # How many distinct continuations the log took after this prefix. A property of the log rather
    # than of the model, so it is the same for every model of a log.
    reference_size: float = metric(unit=Unit.COUNT, owner=Owner.LOG)

    # How many times the log ran this prefix at all, which is the sample size every score above
    # was read against and so what says whether the prefix is one of the ones they are read over.
    # A property of the log, like the two above it.
    reference_occurrences: float = metric(unit=Unit.COUNT, owner=Owner.LOG)

    @classmethod
    def of(cls, generation: Generation, *, index: ContinuationIndex) -> Self:
        """Score the suffixes generated for one prefix against every suffix observed after it.

        Args:
            generation: The model's answer for one prefix, decoded into the log's own units.
            index: The log's observed continuations, held by the process doing the scoring.
        Returns:
            The prefix's scores.
        """
        samples = generation.samples
        references = index.continuations(generation.prefix_activities)

        # The distinct suffixes and how many draws each stands for. The generations are written on
        # the index's own scale, so nothing is encoded here.
        suffixes, counts = samples.suffixes, samples.counts
        draws = len(samples)

        # Comparing a prefix's samples against each other is what measures the spread
        # `p(z | prefix)` claims the prefix leaves open. A suffix drawn twice is twice as likely to
        # be picked and the pair it makes with itself sits at distance 0, which is exactly what
        # `diversity` weighs.
        sample_diversity = diversity(suffixes, weights=counts)

        observed, generated = set(references.suffixes), set(suffixes)
        covered = sum(
            weight
            for suffix, weight in zip(references.suffixes, references.weights, strict=True)
            if suffix in generated
        )

        lengths = [float(len(suffix)) for suffix in suffixes] or [0.0]
        remaining = [events.remaining_time_minutes for events in samples.events] or [0.0]

        # A cycle time belongs to the activity it precedes, so a draw's suffix is read a character
        # at a time against the cycle times that draw was written with. `_decode` cuts a run's
        # activities and its cycle times to one length, so the two always pair up. Read per draw
        # rather than per distinct suffix: two draws of one suffix came from different `z` and carry
        # different cycle times.
        drawn: dict[str, list[float]] = {}
        for events in samples.events:
            for activity, cycle_time in zip(
                events.activities, events.cycle_time_minutes, strict=True
            ):
                drawn.setdefault(activity, []).append(cycle_time)

        return cls(
            emsc=emsc(suffixes=suffixes, counts=counts, references=references),
            continuation_recall=covered / references.occurrences,
            continuation_precision=(
                float(counts @ [float(suffix in observed) for suffix in suffixes]) / draws
                if len(suffixes) and draws
                else 0.0
            ),
            # A continuation's length is fixed by its activities, so the distinct suffixes weighed
            # by their counts are the same distribution as one length per occurrence. That holds on
            # the generated side too, which is why the draws are weighed rather than unfolded.
            length_wasserstein=wasserstein_distance(
                u_values=lengths,
                u_weights=counts if len(suffixes) else None,
                v_values=[float(len(suffix)) for suffix in references.suffixes],
                v_weights=references.weights,
            ),
            remaining_time_wasserstein_days=wasserstein_distance(
                u_values=remaining,
                v_values=references.remaining_times,
            )
            / MINUTES_PER_DAY,
            activity_time_wasserstein_days=activity_time_wasserstein_minutes(
                generated=drawn, observed=references.cycle_times
            )
            / MINUTES_PER_DAY,
            sample_diversity=sample_diversity,
            unique_sample_rate=len(suffixes) / draws if draws else 0.0,
            reference_diversity=references.diversity,
            reference_size=float(len(references.suffixes)),
            reference_occurrences=references.occurrences,
        )

    @property
    def comparable(self) -> bool:
        """Whether the log ran this prefix often enough for its scores to be read.

        Every score of this family compares two distributions, and one of them is estimated
        from the occurrences of the prefix alone. A prefix the log ran once leaves an estimate
        that is a single point, against which each of these scores is an accuracy under another
        name and is maximized by drawing one suffix every time. Such a prefix is still scored
        and still written to the per-prefix file, since the numbers are read there against the
        prefix that carries them; it is the means that are taken over the prefixes this admits.

        Returns:
            Whether the log ran the prefix at least `MIN_REFERENCE_OCCURRENCES` times.
        """
        return self.reference_occurrences >= MIN_REFERENCE_OCCURRENCES

    @classmethod
    def undefined(cls) -> Self:
        """What a set of prefixes none of which is comparable scores, which is nothing.

        NaN throughout rather than the 0.0 an empty mean would give: 0.0 is the worst several of
        these can be and would read as a model that scored nothing where the truth is that the
        length was never measured. Every reader here already takes NaN for a metric that has no
        mean to report.

        Returns:
            Every field NaN.
        """
        return cls(**{entry.name: float('nan') for entry in fields(cls)})


def emsc(suffixes: tuple[str, ...], counts: np.ndarray, references: Continuations) -> float:
    """Earth Movers' Stochastic Conformance between generated and observed continuations.

    The two sets of suffixes are read as stochastic languages, the generated one uniform over the
    draws and the observed one weighted by how often each continuation occurred, and compared by
    the optimal transport cost of turning the first into the second under the normalized
    Damerau-Levenshtein distance. Reported as a similarity so it reads the way every other share
    here does.

    Solved over every continuation the prefix was observed to take. The short prefixes of a large
    log run to a few thousand of them, which the exact solver handles in tens of milliseconds, and
    dropping the light tail is not free: those prefixes are the ones a log revisits most, so an
    approximation there would move the mean of a tenth of the split.

    Both sides are held as distinct sequences carrying the mass of the draws or occurrences they
    stand for, which is the same stochastic language a row per draw would spell out and a smaller
    problem to solve: a run whose draws collapse onto a handful of suffixes gets a matrix that many
    rows tall rather than one row per draw.

    Args:
        suffixes: The distinct generated suffixes, on the index's scale.
        counts: How many draws took each of them, in the same order.
        references: The prefix's observed continuations, their weights and their remaining times.
    Returns:
        `1 - cost` in `[0, 1]`, 1.0 where the model reproduces the prefix's continuations exactly
        and 0.0 where no draw shares anything with any of them. 0.0 for a prefix with no draws,
        the worst it can be, rather than looking like a perfect match.
    """
    if not suffixes:
        return 0.0

    # `[len(suffixes), len(references.suffixes)]`, one row per distinct draw and one column per
    # distinct continuation.
    cost = distances(queries=suffixes, choices=references.suffixes, dtype=np.float64)
    drawn = counts / counts.sum()
    weights = references.weights / references.occurrences
    return 1.0 - float(ot.emd2(a=drawn, b=weights, M=cost))


def activity_time_wasserstein_minutes(
    generated: Mapping[str, Sequence[float]], observed: Mapping[str, np.ndarray]
) -> float:
    """How far the cycle times a model puts before each activity are from the ones the log put
    there.

    One 1-Wasserstein distance per activity, between the cycle times pooled over every draw and the
    ones pooled over every occurrence of the prefix, averaged with each activity weighed by how
    often the log ran it. Grouping by the activity is what makes each of them a comparison of two
    conditional distributions, so a draw running longer or shorter than an occurrence normalizes
    away rather than leaking into the timing. Read by position instead, one inserted event shifts
    every cycle time after it and the number reports as a timing error what the activities were
    wrong about.

    An activity only one side ran is skipped. Writing an activity the log never took after this
    prefix, or never writing one it did, is a control-flow error, and `emsc`,
    `continuation_precision` and `continuation_recall` are what charge for it; counting it here
    would restate it as a timing error, which is the same mistake reading the cycle times by
    position makes.

    Both sides are small, so this is biased upward the way `length_wasserstein` and
    `remaining_time_wasserstein_days` are. The bias follows the draw count and the prefix's
    occurrence count, both of which every model of a log shares, so it is a number to compare
    models on rather than a distance to quote on its own.

    Args:
        generated: The cycle times of every draw, pooled under the activity each of them precedes.
        observed: The same over every continuation the prefix was observed to take.
    Returns:
        The weighted mean distance, in minutes. Where the two share no activity the pools are
        compared unconditioned instead, since 0.0 would read as a perfect match on a score that
        has no worst value of its own.
    """
    scored = [
        (
            float(len(cycle_times)),
            wasserstein_distance(u_values=generated[activity], v_values=cycle_times),
        )
        for activity, cycle_times in observed.items()
        if activity in generated
    ]
    if scored:
        return sum(weight * distance for weight, distance in scored) / sum(
            weight for weight, _ in scored
        )
    return wasserstein_distance(
        u_values=[cycle_time for cycle_times in generated.values() for cycle_time in cycle_times]
        or [0.0],
        v_values=[cycle_time for cycle_times in observed.values() for cycle_time in cycle_times],
    )
