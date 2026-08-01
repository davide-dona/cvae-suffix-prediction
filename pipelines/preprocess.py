import argparse
from pathlib import Path
import pandas as pd

from src.configs import DataConfig, load_config
from src.logs.io import read_log, write_log
from src.logs.keys import (
    CASE_OFFSET_KEY,
    EVENT_DELTA_KEY,
    ACTIVITY_KEY,
    CASE_KEY,
    LABEL_KEY,
    REMAINING_TIME_KEY,
    RESOURCE_KEY,
    TIMESTAMP_KEY,
)
from src.logs.discovery import discover_process_model, write_process_model
from src.logs.split import temporal_split
from src.logs.timestamps import add_case_offset, add_event_delta, add_remaining_time


def preprocess(log: pd.DataFrame) -> pd.DataFrame:
    """
    Add relative-timestamp columns to an event log.
    Args:
        log: Event log with columns already renamed to canonical names
            (see `src.logs.io.read_log`).
    Returns:
        A copy of `log` with `CASE_OFFSET_KEY`, `EVENT_DELTA_KEY` and
        `REMAINING_TIME_KEY` added.
    """
    log = add_case_offset(
        log,
        case_key=CASE_KEY,
        timestamp_key=TIMESTAMP_KEY,
        out_key=CASE_OFFSET_KEY,
    )
    log = add_event_delta(
        log,
        case_key=CASE_KEY,
        timestamp_key=TIMESTAMP_KEY,
        out_key=EVENT_DELTA_KEY,
    )
    log = add_remaining_time(
        log,
        case_key=CASE_KEY,
        timestamp_key=TIMESTAMP_KEY,
        out_key=REMAINING_TIME_KEY,
    )
    return log


def ensure_dataset(data_config: DataConfig) -> None:
    """
    Preprocess the dataset if its splits are not on disk yet, so a training run only ever
    needs the raw log to have been placed in `<dir>/original.csv`.
    Args:
        data_config: The `data` section of this dataset's experiment config.
    Raises:
        FileNotFoundError: If the splits are missing and there is no raw log to build them from.
    """
    processed_dir = data_config.dir / 'processed'
    if all((processed_dir / f'{split}.csv').exists() for split in ('train', 'val', 'test')):
        return

    original = data_config.dir / 'original.csv'
    if not original.exists():
        raise FileNotFoundError(
            f'{processed_dir} has no train/val/test splits and there is no raw log at {original} '
            'to build them from.'
        )

    print(f'Splits missing under "{processed_dir}", preprocessing "{original}" first')
    run(data_config)


def run(data_config: DataConfig) -> None:
    """
    Preprocess and split a dataset, writing outputs next to the input.

    Reads `<dir>/original.csv`, renames its structural columns to the
    canonical names used throughout the codebase, adds the relative-timestamp
    columns, writes the result as `<dir>/processed/full.csv`, then splits it by
    case start time into `train.csv`, `val.csv` and `test.csv` in the same folder.

    Args:
        data_config: The `data` section of this dataset's experiment config.
    """
    column_mapping = {
        data_config.case_key: CASE_KEY,
        data_config.activity_key: ACTIVITY_KEY,
        data_config.resource_key: RESOURCE_KEY,
        data_config.timestamp_key: TIMESTAMP_KEY,
        data_config.label_key: LABEL_KEY,
    }

    # Read, preprocess and write the full elaborated log\
    log = read_log(data_config.dir / 'original.csv', column_mapping=column_mapping)
    log = preprocess(log)
    processed_dir = data_config.dir / 'processed'
    write_log(log, processed_dir / 'full.csv')

    # Split the log into train/val/test and write them. Test is whatever the first two fractions
    # leave over, so `data_config.test_split` is not passed
    train, val, test = temporal_split(
        log,
        case_key=CASE_KEY,
        timestamp_key=TIMESTAMP_KEY,
        train_frac=data_config.train_split,
        val_frac=data_config.val_split,
    )
    write_log(train, processed_dir / 'train.csv')
    write_log(val, processed_dir / 'val.csv')
    write_log(test, processed_dir / 'test.csv')

    # Discover a Petri net of the process and write it to disk. 
    # Done using the inductive miner, only on the training set to avoid data leakage.
    net, initial_marking, final_marking = discover_process_model(
        train,
        case_key=CASE_KEY,
        activity_key=ACTIVITY_KEY,
        timestamp_key=TIMESTAMP_KEY,
    )
    write_process_model(net, initial_marking, final_marking, data_config.dir / 'model')

    print(f'Preprocessed "{data_config.dir}": {len(train)} train, {len(val)} val, {len(test)} test events')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Turn a raw event log into the train/val/test CSVs the model consumes.'
    )
    parser.add_argument('-c', '--config', type=Path, required=True,
                         help="Path to this dataset's experiment config YAML.")
    args = parser.parse_args()

    config = load_config(args.config)
    run(config.data)


if __name__ == '__main__':
    main()
