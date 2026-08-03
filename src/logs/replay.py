from typing import Sequence
import pandas as pd
import pm4py
from pm4py.objects.petri_net.obj import Marking, PetriNet

from src.logs.keys import ACTIVITY_KEY, CASE_KEY, TIMESTAMP_KEY


def replay_fitness(
    traces: Sequence[Sequence[str]],
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
) -> list[float]:
    """Token-based replay fitness of each trace against a Petri net.
    A fitness of 1.0 means the trace is perfectly replayable, 0.0 means it is not replayable at all.
    
    Each distinct trace is replayed once, and the result is scattered back to every instance of it in the input.

    Args:
        traces: The traces to replay, each a sequence of activity names.
        net: The Petri net to replay against.
        initial_marking: Its initial marking.
        final_marking: Its final marking.
    Returns:
        One fitness per input trace, in input order. A trace scores 0.0 without being replayed
        if it is empty, or if it names an activity the net has no transition for: pm4py steps
        over such an event as though it were not there, which would score a generated `UNK` as
        a perfect fit.
    """
    # Retrieve the set of activity labels from the net
    labels = {transition.label for transition in net.transitions}

    # Group the traces by their distinct activity sequences, so that each distinct trace is replayed once
    variants: dict[tuple[str, ...], list[int]] = {}     # (distinct trace, positions in `traces` where it occurs)
    for position, trace in enumerate(traces):
        if trace and labels.issuperset(trace):
            variants.setdefault(tuple(trace), []).append(position)

    fitness = [0.0] * len(traces)
    if not variants:
        return fitness

    # Replay the distinct traces
    diagnostics = pm4py.conformance_diagnostics_token_based_replay(
        _as_log(list(variants)),
        net,
        initial_marking,
        final_marking,
        activity_key=ACTIVITY_KEY,
        timestamp_key=TIMESTAMP_KEY,
        case_id_key=CASE_KEY,
        opt_parameters={'show_progress_bar': False},
    )
    # Scatter the fitness of each distinct trace back to every instance of it in the input
    for diagnosis, positions in zip(diagnostics, variants.values()):
        for position in positions:
            fitness[position] = diagnosis['trace_fitness']

    return fitness


def replay_precision(
    traces: Sequence[Sequence[str]],
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
) -> float:
    """ET-Conformance token-based precision of a set of traces against a Petri net.

    Precision is a whole-set measure, not a per-trace one: it is the fraction of transitions
    the net enables at each prefix the traces actually visit that the traces go on to take,
    weighted by how often each prefix recurs. A permissive net that still replays every trace
    at fitness 1.0 (as the inductive miner guarantees for its own training log) shows up here
    instead, since it keeps offering transitions the traces never use.

    Args:
        traces: The traces to score, each a sequence of activity names. Not deduplicated:
            how often a prefix recurs is part of what precision weighs.
        net: The Petri net to score against.
        initial_marking: Its initial marking.
        final_marking: Its final marking.
    Returns:
        The precision of the traces naming only activities the net has a transition for, 1.0
        if none do (pm4py's convention for an empty log, kept so an all-unknown set reads as
        uninformative rather than as a fitness-like 0.0).
    """
    labels = {transition.label for transition in net.transitions}
    known = [trace for trace in traces if trace and labels.issuperset(trace)]
    if not known:
        return 1.0

    return pm4py.precision_token_based_replay(
        _as_log(known),
        net,
        initial_marking,
        final_marking,
        activity_key=ACTIVITY_KEY,
        timestamp_key=TIMESTAMP_KEY,
        case_id_key=CASE_KEY,
    )


def _as_log(traces: Sequence[Sequence[str]]) -> pd.DataFrame:
    """Turn activity sequences into the flat event log pm4py replays.

    Case ids are zero-padded so that sorting them lexicographically, which is what pm4py does
    when it groups the frame into traces, restores the order the traces came in; the
    timestamps are synthetic and only serve to order the events within a trace.

    Args:
        traces: The traces to convert, each a sequence of activity names. Duplicates are kept
            as separate cases: callers that want each distinct trace replayed once dedupe
            before calling.
    Returns:
        One row per event, with the canonical case/activity/timestamp columns.
    """
    width = len(str(len(traces) - 1))
    epoch = pd.Timestamp(ts_input='2000-01-01')
    events = [
        {
            CASE_KEY: f'{index:0{width}d}',
            ACTIVITY_KEY: activity,
            TIMESTAMP_KEY: epoch + pd.Timedelta(minutes=position),
        }
        for index, trace in enumerate(traces)
        for position, activity in enumerate(trace)
    ]
    return pd.DataFrame(events)
