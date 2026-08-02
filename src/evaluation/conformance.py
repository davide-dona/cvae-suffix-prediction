from dataclasses import dataclass
from typing import Sequence
import pandas as pd
from pm4py.objects.petri_net.obj import Marking, PetriNet

from src.evaluation.sequences import mean
from src.logs.keys import ACTIVITY_KEY, CASE_KEY
from src.logs.replay import replay_fitness


@dataclass(frozen=True)
class TraceConformance:
    """How much of a set of traces the process allows."""
    fitness_mean: float
    perfectly_fitting_rate: float
    unseen_bigram_rate: float


@dataclass(frozen=True)
class ConformanceMetrics:
    """The generated traces' conformance, and the same numbers for the real ones.

    The reference is not decoration. The net is mined by the inductive miner and drops
    behaviour, so the log it was mined from does not score 1.0 either; a generated fitness only
    means something as a fraction of what the ground truth reaches on the same prefixes.
    """
    generated: TraceConformance
    reference: TraceConformance


def conformance_metrics(
    generations: pd.DataFrame,
    *,
    test_log: pd.DataFrame,
    train_log: pd.DataFrame,
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
) -> ConformanceMetrics:
    """
    Replay the generated suffixes against the process model mined from the training split.

    What is replayed is the whole trace, prefix followed by suffix, not the suffix on its own:
    a suffix starts wherever its prefix left the net, so replaying it from the initial marking
    would score the prefix's absence rather than the suffix's quality.

    Args:
        generations: Rows written by `src/inference/generate.py`, truncated pairs already
            dropped, one per (prefix, sample).
        test_log: The split the prefixes were cut from, supplying the events the rows only
            record a length for.
        train_log: The split the net was mined from, supplying the transitions ever observed.
        net: The mined Petri net.
        initial_marking: Its initial marking.
        final_marking: Its final marking.
    Returns:
        The generated traces' conformance and the ground truth's.
    """
    prefixes = _case_activities(test_log)
    observed_bigrams = set(_bigrams(list(_case_activities(train_log).values())))

    generated_prefixes: list[list[str]] = []
    generated_suffixes: list[list[str]] = []
    for row in generations.itertuples():
        generated_prefixes.append(prefixes[row.case_id][:row.prefix_len])
        generated_suffixes.append(list(row.generated_activities))

    # One ground truth per prefix, however many samples were drawn from it.
    truths = generations.drop_duplicates(subset=['case_id', 'prefix_len'])
    reference_prefixes: list[list[str]] = []
    reference_suffixes: list[list[str]] = []
    for row in truths.itertuples():
        reference_prefixes.append(prefixes[row.case_id][:row.prefix_len])
        reference_suffixes.append(list(row.true_activities))

    return ConformanceMetrics(
        generated=_conformance(
            prefixes=generated_prefixes,
            suffixes=generated_suffixes,
            observed_bigrams=observed_bigrams,
            net=net,
            initial_marking=initial_marking,
            final_marking=final_marking,
        ),
        reference=_conformance(
            prefixes=reference_prefixes,
            suffixes=reference_suffixes,
            observed_bigrams=observed_bigrams,
            net=net,
            initial_marking=initial_marking,
            final_marking=final_marking,
        ),
    )


def _conformance(
    *,
    prefixes: Sequence[Sequence[str]],
    suffixes: Sequence[Sequence[str]],
    observed_bigrams: set[tuple[str, str]],
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
) -> TraceConformance:
    """Score one set of (prefix, suffix) pairs.

    Args:
        prefixes: The prefix each suffix continues, in the same order as `suffixes`.
        suffixes: The suffixes to score.
        observed_bigrams: Every directly-follows pair of the training split.
        net, initial_marking, final_marking: The mined process model.
    Returns:
        The three conformance numbers for this set.
    """
    traces = [list(prefix) + list(suffix) for prefix, suffix in zip(prefixes, suffixes)]
    fitness = replay_fitness(traces, net, initial_marking, final_marking)

    # Only the pairs the suffix introduces, the one joining it to its prefix included. The
    # prefix's own pairs come from the log and are observed by construction.
    introduced = _bigrams(
        [list(prefix[-1:]) + list(suffix) for prefix, suffix in zip(prefixes, suffixes)]
    )
    unseen = [bigram for bigram in introduced if bigram not in observed_bigrams]

    return TraceConformance(
        fitness_mean=mean(fitness),
        perfectly_fitting_rate=mean([float(f >= 1.0) for f in fitness]),
        # Pooled over pairs rather than averaged over traces, so a long suffix's transitions
        # weigh what they are: one chance each to leave the observed behaviour.
        unseen_bigram_rate=len(unseen) / len(introduced) if introduced else 0.0,
    )


def _case_activities(log: pd.DataFrame) -> dict[str, list[str]]:
    """The activity sequence of every case, in event order.

    Args:
        log: A processed split, read by `src.logs.io.read_log`.
    Returns:
        `{case id: activities}`.
    """
    grouped = log.groupby(CASE_KEY, sort=False)[ACTIVITY_KEY]
    return {str(case_id): activities.tolist() for case_id, activities in grouped}


def _bigrams(traces: Sequence[Sequence[str]]) -> list[tuple[str, str]]:
    """Every directly-follows pair of a collection of traces, occurrences included.

    Args:
        traces: The traces to read pairs off.
    Returns:
        One pair per pair of consecutive events, in trace order.
    """
    return [
        (str(trace[i]), str(trace[i + 1]))
        for trace in traces
        for i in range(len(trace) - 1)
    ]
