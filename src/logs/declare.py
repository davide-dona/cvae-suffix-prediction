from collections.abc import Sequence

import pandas as pd
import pm4py
from Declare4Py.D4PyEventLog import D4PyEventLog
from Declare4Py.ProcessMiningTasks.Discovery.DeclareMiner import DeclareMiner
from Declare4Py.ProcessModels.DeclareModel import DeclareModel
from Declare4Py.Utils.Declare.Checkers import ConstraintChecker
from Declare4Py.Utils.Declare.TraceStates import TraceState

from src import paths
from src.configs import DeclareConfig
from src.logs.keys import ACTIVITY_KEY, CASE_KEY, TIMESTAMP_KEY


def discover_declare_model(
    train: pd.DataFrame,
    *,
    dataset: str,
    declare_config: DeclareConfig,
) -> int:
    """
    Discover a declarative model from the train split and write it beside the dataset.

    Args:
        train: The train split, as preprocessing holds it. The only log discovery reads, so the
            constraints never carry anything from the validation or test split.
        dataset: The dataset the split came from, naming where the model goes.
        declare_config: The `declare` section: which constraints are looked for and how much of
            the log has to support one.

    Returns:
        The number of constraints written.
    """
    # Drop any columns that are not the structural ones
    event_log = pm4py.convert_to_event_log(train[[CASE_KEY, ACTIVITY_KEY, TIMESTAMP_KEY]])
    # Converting from a DataFrame leaves these unset, and D4PyEventLog reads them.
    event_log.properties['pm4py:param:activity_key'] = ACTIVITY_KEY
    event_log.properties['pm4py:param:timestamp_key'] = TIMESTAMP_KEY

    miner = DeclareMiner(
        log=D4PyEventLog(case_name=CASE_KEY, log=event_log),
        consider_vacuity=declare_config.consider_vacuity,
        min_support=declare_config.min_support,
        itemsets_support=declare_config.itemsets_support,
        max_declare_cardinality=declare_config.max_cardinality,
    )
    model = miner.run()

    # Write the activities in the Declare4py format
    lines = [f'activity {activity}' for activity in model.activities]
    # Write the constraints in the Declare4py format, one per line
    for constraint, serialized in zip(model.constraints, model.serialized_constraints, strict=True):
        # Declare4Py serializes a binary constraint's two conditions as the one empty field a
        # unary constraint gets, which its own parser then rejects; the missing separator is
        # added back here.
        lines.append(f'{serialized} |' if constraint['template'].is_binary else serialized)

    path = paths.declare_model_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')

    return len(model.constraints)


def load_declare_model(dataset: str) -> DeclareModel:
    """Read back the model discovered at preprocessing time.

    Args:
        dataset: The dataset whose model to read.
    Returns:
        The parsed model, ready for `conformance_rate`.
    """
    return DeclareModel().parse_from_file(str(paths.declare_model_path(dataset)))


def conformance_rate(
    activities: Sequence[str],
    *,
    model: DeclareModel,
    consider_vacuity: bool,
) -> float:
    """
    The fraction of a declarative model's constraints one trace satisfies.

    Args:
        activities: The trace's activity names, in order. A whole case, prefix included: a
            constraint like `Init` or `Precedence` is about the trace, not about a run of events
            inside it.
        model: The model to check against, from `load_declare_model`.
        consider_vacuity: Whether a constraint the trace never activates counts as satisfied.
            False makes it count as violated instead, so the rate only credits constraints the
            trace actually exercises.
    Returns:
        The satisfied share, in `[0, 1]`, or 0.0 for a model that checked nothing.
    """
    # The checkers only ever read `event[ACTIVITY_KEY]` and the trace's length, so this is all a
    # trace has to be: no event log, and nothing read off disk per call.
    trace = [{ACTIVITY_KEY: activity} for activity in activities]
    # Every trace is judged as a finished case, which `check_trace_conformance` assumes anyway.
    results = ConstraintChecker().check_trace_conformance(
        trace, model, consider_vacuity, ACTIVITY_KEY
    )
    if not results:
        return 0.0
    # A constraint whose conditions fail to parse is dropped by the checker, so the results are
    # the denominator rather than the model's constraints.
    satisfied = sum(result.state == TraceState.SATISFIED for result in results)
    return satisfied / len(results)
