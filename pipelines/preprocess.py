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
    RESOURCE_KEY,
    TIMESTAMP_KEY,
)
from src.logs.split import temporal_split
from src.logs.timestamps import add_case_offset, add_event_delta


def preprocess(log: pd.DataFrame) -> pd.DataFrame:
    """
    Add the relative-timestamp columns the model consumes.
    Args:
        log: Event log with columns already renamed to canonical names
            (see `src.logs.io.read_log`).
    Returns:
        A copy of `log` with `CASE_OFFSET_KEY` and
        `EVENT_DELTA_KEY` added.
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
    return log


def run(data_config: DataConfig) -> None:
    """
    Preprocess and split a dataset, writing outputs next to the input.

    Reads `<dir>/original.csv`, renames its structural columns to the
    canonical names used throughout the codebase, adds the relative-timestamp
    columns, writes the result as `<dir>/full.csv`, then splits it by case
    start time into `train.csv`, `val.csv` and `test.csv` in the same folder.

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

    # Read, preprocess and write the full log
    log = read_log(data_config.dir / 'original.csv', column_mapping=column_mapping)
    log = preprocess(log)
    write_log(log, data_config.dir / 'full.csv')

    # Split the log into train/val/test and write them
    train, val, test = temporal_split(
        log,
        case_key=CASE_KEY,
        timestamp_key=TIMESTAMP_KEY,
        train_frac=data_config.train_split,
        val_frac=data_config.val_split,
        test_frac=data_config.test_split,
    )
    write_log(train, data_config.dir / 'train.csv')
    write_log(val, data_config.dir / 'val.csv')
    write_log(test, data_config.dir / 'test.csv')

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
