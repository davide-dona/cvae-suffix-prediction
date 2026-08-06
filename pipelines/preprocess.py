import argparse
from pathlib import Path
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.configs import DataConfig, DeclareConfig, load_config
from src.datasets.description import DatasetDescription, metadata_path
from src.logs.declare import declare_model_path, discover_declare_model
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
from src.logs.split import temporal_split
from src.logs.timestamps import add_case_offset, add_event_delta, add_remaining_time


def preprocess(log: pd.DataFrame, *, feature_columns: list[str]) -> pd.DataFrame:
    """Preprocess an event log for model training.
    
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


def require_dataset(data_config: DataConfig) -> None:
    """Check that everything preprocessing produces is on disk, and say what is missing if it is not.

    Args:
        data_config: The `data` section of this dataset's experiment config.
    Raises:
        FileNotFoundError: If any preprocessing output is missing, naming every one of them.
    """
    processed_dir = data_config.dir / 'processed'
    outputs = [processed_dir / f'{split}.csv' for split in ('train', 'val', 'test')]
    outputs.append(metadata_path(data_config.dir))
    outputs.append(declare_model_path(data_config))

    missing = [output for output in outputs if not output.exists()]
    if missing:
        raise FileNotFoundError(
            f'"{data_config.dir}" has not been preprocessed: '
            f'{", ".join(str(output) for output in missing)} '
            f'{"are" if len(missing) > 1 else "is"} missing. Run '
            f'"uv run python -m pipelines.preprocess -c <config>" first.'
        )


def run(data_config: DataConfig, declare_config: DeclareConfig) -> None:
    """
    Preprocess and split a dataset, writing outputs next to the input.

    Reads `<dir>/original.csv`, renames its structural columns to the
    canonical names used throughout the codebase, extract additional features,
    and splits the log into train/val/test. 

    The vocabularies and normalization ranges the model is built against are fit here too,
    on the train split alone, and written beside it as `dataset.json`. The declarative model is
    discovered from the same split and written to `<dir>/declare/model.decl`.

    Args:
        data_config: The `data` section of this dataset's experiment config.
        declare_config: The `declare` section, driving the discovery of the declarative model.
    """
    column_mapping = {
        data_config.case_key: CASE_KEY,
        data_config.activity_key: ACTIVITY_KEY,
        data_config.resource_key: RESOURCE_KEY,
        data_config.timestamp_key: TIMESTAMP_KEY,
        data_config.label_key: LABEL_KEY,
    }

    # Read, preprocess and write the full elaborated log
    print(f'Preprocessing "{data_config.dir}"...')
    log = read_log(data_config.dir / 'original.csv', column_mapping=column_mapping)
    log = preprocess(log, feature_columns=data_config.event_features)
    processed_dir = data_config.dir / 'processed'
    write_log(log, processed_dir / 'full.csv')

    # Split the log into train/val/test and write them out
    print(f'Splitting "{data_config.dir}" into train/val/test...')
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

    # Fit the vocabularies and normalization ranges on the train split, writng them out to `dataset.json`
    # The generated values can be decoded back using the same description.
    description = DatasetDescription.fit(train, data_config=data_config)
    description.save()

    # Discover the declarative model from the train split and write it out to `model.decl`.
    num_constraints = discover_declare_model(
        train,
        data_config=data_config,
        declare_config=declare_config,
    )

    print(
        f'Preprocessed "{data_config.dir}": {len(train)} train, {len(val)} val, {len(test)} test '
        f'events, {len(description.activity.vocab)} activities, '
        f'{len(description.resource.vocab)} resources, '
        f'{len(description.categorical_features)} categorical and '
        f'{len(description.numeric_features)} numeric feature channels, '
        f'{num_constraints} declarative constraints'
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Turn a raw event log into the train/val/test CSVs the model consumes.'
    )
    parser.add_argument('-c', '--config', type=Path, required=True,
                         help="Path to this dataset's experiment config YAML.")
    args = parser.parse_args()

    config = load_config(args.config)
    run(data_config=config.data, declare_config=config.declare)


if __name__ == '__main__':
    main()
