from pathlib import Path
import pandas as pd
import pm4py
from pm4py.objects.petri_net.obj import Marking, PetriNet


def discover_process_model(
    log: pd.DataFrame,
    *,
    case_key: str,
    activity_key: str,
    timestamp_key: str,
) -> tuple[PetriNet, Marking, Marking]:
    """
    Mine a Petri net of the process behind a log with the inductive miner.
    Args:
        log: Event log, one row per event, with canonical column names already applied.
        case_key: Column identifying the case each event belongs to.
        activity_key: Column identifying the activity label.
        timestamp_key: Column holding the (already parsed) event timestamp.
    Returns:
        `(net, initial_marking, final_marking)`.
    """
    return pm4py.discover_petri_net_inductive(
        log,
        case_id_key=case_key,
        activity_key=activity_key,
        timestamp_key=timestamp_key,
    )


def write_process_model(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    dir: str | Path,
) -> Path:
    """
    Write the Petri net to disk in PNML format and as a PNG image.
    Args:
        net: The Petri net.
        initial_marking: Its initial marking.
        final_marking: Its final marking.
        dir: Destination directory. Parent directories are created if missing.
    """
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    pm4py.write_pnml(net, initial_marking, final_marking, str(dir / 'model.pnml'))
    pm4py.save_vis_petri_net(net, initial_marking, final_marking, str(dir / 'model.png'))


def read_process_model(dir: str | Path) -> tuple[PetriNet, Marking, Marking]:
    """
    Read back the Petri net `write_process_model` wrote.

    Args:
        dir: The directory holding `model.pnml`.
    Returns:
        `(net, initial_marking, final_marking)`.
    Raises:
        FileNotFoundError: If no net has been mined for this dataset yet.
    """
    path = Path(dir) / 'model.pnml'
    if not path.exists():
        raise FileNotFoundError(
            f'no process model at {path}. Preprocess the dataset first, which mines one '
            'from its training split.'
        )
    return pm4py.read_pnml(str(path))
