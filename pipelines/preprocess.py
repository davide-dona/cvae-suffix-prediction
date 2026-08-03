import argparse
from pathlib import Path
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.configs import DataConfig, load_config
from src.datasets.description import DatasetDescription, metadata_path
from src.logs.io import read_log, write_log
from src.logs.keys import (
    CASE_OFFSET_KEY,
    EVENT_DELTA_KEY,
    ACTIVITY_KEY,
    CASE_KEY,
    LABEL_KEY,
    MISSING_FEATURE,
    REMAINING_TIME_KEY,
    RESOURCE_KEY,
    TIMESTAMP_KEY,
)
from src.logs.discovery import discover_process_model, write_process_model
from src.logs.split import temporal_split
from src.logs.timestamps import add_case_offset, add_event_delta, add_remaining_time


def preprocess(log: pd.DataFrame, *, feature_columns: list[str]) -> pd.DataFrame:
    """
    Add relative-timestamp columns to an event log and write its missing categories out.
    Args:
        log: Event log with columns already renamed to canonical names
            (see `src.logs.io.read_log`).
        feature_columns: `data.event_features`. The non-numeric ones are the categorical
            channels, and their gaps become `MISSING_FEATURE` here, so a value the log does not
            have is a value of its own from this point on rather than something every reader has
            to fill in again. The numeric ones keep their gaps: a missing number is carried by
            the present flag instead.
    Returns:
        A copy of `log` with `CASE_OFFSET_KEY`, `EVENT_DELTA_KEY` and
        `REMAINING_TIME_KEY` added, and its categorical feature columns filled.
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
    # Filling leaves these columns as strings, so the same test in `DatasetDescription.fit` sorts them
    # into the same channels it would have before.
    for column in feature_columns:
        if not is_numeric_dtype(log[column]):
            log[column] = log[column].fillna(MISSING_FEATURE).astype(str)
    return log


def ensure_dataset(data_config: DataConfig) -> None:
    """
    Preprocess the dataset if its splits and description are not on disk yet, so a training run
    only ever needs the raw log to have been placed in `<dir>/original.csv`.
    Args:
        data_config: The `data` section of this dataset's experiment config.
    Raises:
        FileNotFoundError: If the outputs are missing and there is no raw log to build them from.
    """
    processed_dir = data_config.dir / 'processed'
    outputs = [processed_dir / f'{split}.csv' for split in ('train', 'val', 'test')]
    outputs.append(metadata_path(data_config))
    if all(output.exists() for output in outputs):
        return

    original = data_config.dir / 'original.csv'
    if not original.exists():
        raise FileNotFoundError(
            f'{processed_dir} is missing splits or a dataset description, and there is no raw '
            f'log at {original} to build them from.'
        )

    print(f'Outputs missing under "{processed_dir}", preprocessing "{original}" first')
    run(data_config)


def run(data_config: DataConfig) -> None:
    """
    Preprocess and split a dataset, writing outputs next to the input.

    Reads `<dir>/original.csv`, renames its structural columns to the
    canonical names used throughout the codebase, adds the relative-timestamp
    columns, writes the result as `<dir>/processed/full.csv`, then splits it by
    case start time into `train.csv`, `val.csv` and `test.csv` in the same folder.

    The vocabularies and normalization ranges the model is built against are fit here too,
    on the train split alone, and written beside it as `dataset.json`.

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
    log = preprocess(log, feature_columns=data_config.event_features)
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

    # Fit the vocabularies and normalization ranges on the train split and write them beside it,
    # so every later run reads one description rather than deriving its own.
    description = DatasetDescription.fit(train, data_config=data_config)
    description.save(data_config)

    # Discover a Petri net of the process and write it to disk. 
    # Done using the inductive miner, only on the training set to avoid data leakage.
    net, initial_marking, final_marking = discover_process_model(
        train,
        case_key=CASE_KEY,
        activity_key=ACTIVITY_KEY,
        timestamp_key=TIMESTAMP_KEY,
    )
    write_process_model(net, initial_marking, final_marking, data_config.dir / 'model')

    print(
        f'Preprocessed "{data_config.dir}": {len(train)} train, {len(val)} val, {len(test)} test '
        f'events, {len(description.activity.vocab)} activities, '
        f'{len(description.resource.vocab)} resources, '
        f'{len(description.categorical_features)} categorical and '
        f'{len(description.numeric_features)} numeric feature channels'
    )


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
