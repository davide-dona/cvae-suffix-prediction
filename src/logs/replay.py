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
    """
    Token-based replay fitness of each trace against a Petri net.

    A fitness of 1.0 means the trace can be replayed from the initial marking to the final one
    without a token ever being missing or left behind; anything less is behaviour the model
    does not allow. Traces are activity sequences only: token replay reads nothing else.

    Distinct traces are replayed once each and the result is scattered back, since generated
    suffixes repeat heavily and replaying is the expensive part of evaluation.

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
    labels = {transition.label for transition in net.transitions}

    variants: dict[tuple[str, ...], list[int]] = {}
    for position, trace in enumerate(traces):
        if trace and labels.issuperset(trace):
            variants.setdefault(tuple(trace), []).append(position)

    fitness = [0.0] * len(traces)
    if not variants:
        return fitness

    diagnostics = pm4py.conformance_diagnostics_token_based_replay(
        _as_log(list(variants)),
        net,
        initial_marking,
        final_marking,
        activity_key=ACTIVITY_KEY,
        timestamp_key=TIMESTAMP_KEY,
        case_id_key=CASE_KEY,
        # pm4py draws a progress bar per call otherwise, which says nothing a pipeline's own
        # output does not already say.
        opt_parameters={'show_progress_bar': False},
    )
    for diagnosis, positions in zip(diagnostics, variants.values()):
        for position in positions:
            fitness[position] = diagnosis['trace_fitness']

    return fitness


def _as_log(variants: Sequence[tuple[str, ...]]) -> pd.DataFrame:
    """Turn activity sequences into the flat event log pm4py replays.

    Case ids are zero-padded so that sorting them lexicographically, which is what pm4py does
    when it groups the frame into traces, restores the order the variants came in; the
    timestamps are synthetic and only serve to order the events within a trace.

    Args:
        variants: The distinct traces, each a tuple of activity names.
    Returns:
        One row per event, with the canonical case/activity/timestamp columns.
    """
    width = len(str(len(variants) - 1))
    epoch = pd.Timestamp(ts_input='2000-01-01')
    events = [
        {
            CASE_KEY: f'{index:0{width}d}',
            ACTIVITY_KEY: activity,
            TIMESTAMP_KEY: epoch + pd.Timedelta(minutes=position),
        }
        for index, variant in enumerate(variants)
        for position, activity in enumerate(variant)
    ]
    return pd.DataFrame(events)
