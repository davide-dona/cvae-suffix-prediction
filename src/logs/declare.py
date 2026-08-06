from pathlib import Path
import pandas as pd
import pm4py
from Declare4Py.D4PyEventLog import D4PyEventLog
from Declare4Py.ProcessMiningTasks.Discovery.DeclareMiner import DeclareMiner

from src.configs import DataConfig, DeclareConfig
from src.logs.keys import ACTIVITY_KEY, CASE_KEY, TIMESTAMP_KEY


def declare_model_path(data_config: DataConfig) -> Path:
    """Where a dataset's discovered declarative model is kept."""
    return data_config.dir / 'declare' / 'model.decl'


def discover_declare_model(
    train: pd.DataFrame,
    *,
    data_config: DataConfig,
    declare_config: DeclareConfig,
) -> int:
    """
    Discover a declarative model from the train split and write it beside the dataset.

    Args:
        train: The train split, as preprocessing holds it. The only log discovery reads, so the
            constraints never carry anything from the validation or test split.
        data_config: The `data` section, for where the model goes.
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
    for constraint, serialized in zip(model.constraints, model.serialized_constraints):
        # Declare4Py serializes a binary constraint's two conditions as the one empty field a
        # unary constraint gets, which its own parser then rejects; the missing separator is
        # added back here.
        lines.append(f'{serialized} |' if constraint['template'].is_binary else serialized)

    path = declare_model_path(data_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')

    return len(model.constraints)
