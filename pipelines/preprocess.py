import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src import paths
from src.configs import DataConfig, DeclareConfig, load_config
from src.datasets.codec import DatasetCodec
from src.logs.declare import discover_declare_model
from src.logs.io import read_log, write_log
from src.logs.keys import (
    ACTIVITY_KEY,
    CASE_ELAPSED_KEY,
    CASE_KEY,
    DAY_IN_WEEK_KEY,
    EVENT_DELTA_KEY,
    MISSING_FEATURE,
    REMAINING_TIME_KEY,
    RESOURCE_KEY,
    SECONDS_IN_DAY_KEY,
    TIMESTAMP_KEY,
)
from src.logs.split import temporal_split, uedlstm_split
from src.logs.timestamps import (
    add_calendar,
    add_case_elapsed,
    add_event_delta,
    add_remaining_time,
)
from src.paths import Split


def case_length_cutoff(log: pd.DataFrame, *, percentile: float) -> int:
    """Compute the cutoff in events for dropping cases too long to fit the model's sequence tensors.

    Args:
        log: The whole preprocessed log, one row per event, before it is split.
        percentile: `data.max_seq_len_percentile`.
    Returns:
        The cutoff in events: the smallest length at least `percentile` percent of cases fit
        within.
    """
    lengths = log.groupby(CASE_KEY)[CASE_KEY].size()
    return int(np.ceil(np.percentile(lengths, percentile)))


def drop_long_cases(split: pd.DataFrame, *, max_seq_len: int) -> pd.DataFrame:
    """Drop the cases longer than `max_seq_len` events from a split of the log.

    Args:
        split: One split of the log, one row per event.
        max_seq_len: The longest case to keep, in events, from `case_length_cutoff`.
    Returns:
        A row-subset of `split` holding only the cases of at most `max_seq_len` events.
    """
    lengths = split.groupby(CASE_KEY)[CASE_KEY].transform('size')
    return split[lengths <= max_seq_len]


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
        A copy of `log` with the two timestamp proxies, the remaining time and the two calendar
        columns added, and its categorical feature columns filled.
    """
    log = add_event_delta(
        log,
        case_key=CASE_KEY,
        timestamp_key=TIMESTAMP_KEY,
        out_key=EVENT_DELTA_KEY,
    )
    log = add_case_elapsed(
        log,
        case_key=CASE_KEY,
        timestamp_key=TIMESTAMP_KEY,
        out_key=CASE_ELAPSED_KEY,
    )
    log = add_remaining_time(
        log,
        case_key=CASE_KEY,
        timestamp_key=TIMESTAMP_KEY,
        out_key=REMAINING_TIME_KEY,
    )
    log = add_calendar(
        log,
        timestamp_key=TIMESTAMP_KEY,
        day_key=DAY_IN_WEEK_KEY,
        seconds_key=SECONDS_IN_DAY_KEY,
    )
    # Filling leaves these columns as strings, so the same test in `DatasetCodec.fit`
    # sorts them into the same channels it would have before.
    for column in feature_columns:
        if not is_numeric_dtype(log[column]):
            log[column] = log[column].fillna(MISSING_FEATURE).astype(str)
    return log


def require_dataset(dataset: str) -> None:
    """Check that everything preprocessing produces is on disk, and say what is missing if it
    is not.

    Args:
        dataset: The dataset to check.
    Raises:
        FileNotFoundError: If any preprocessing output is missing, naming every one of them.
    """
    outputs = [paths.split_path(dataset=dataset, split=split) for split in Split]
    outputs.append(paths.codec_path(dataset))
    outputs.append(paths.declare_model_path(dataset))

    missing = [output for output in outputs if not output.exists()]
    if missing:
        raise FileNotFoundError(
            f'"{dataset}" has not been preprocessed: '
            f'{", ".join(str(output) for output in missing)} '
            f'{"are" if len(missing) > 1 else "is"} missing. Run '
            f'"uv run python -m pipelines.preprocess -c <config>" first.'
        )


def _split(
    log: pd.DataFrame, *, data_config: DataConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cut a log into train/val/test the way this dataset's config asks for.

    Args:
        log: The preprocessed log, one row per event.
        data_config: The `data` section, naming the strategy and the three proportions.
    Returns:
        `(train, val, test)` DataFrames, each a row-subset of `log`.
    """
    if data_config.split_strategy == 'temporal':
        return temporal_split(
            log,
            case_key=CASE_KEY,
            timestamp_key=TIMESTAMP_KEY,
            train_frac=data_config.train_split,
            val_frac=data_config.val_split,
        )
    return uedlstm_split(
        log,
        case_key=CASE_KEY,
        val_frac=data_config.val_split,
        test_frac=data_config.test_split,
    )


def run(data_config: DataConfig, declare_config: DeclareConfig) -> None:
    """
    Preprocess and split a dataset, writing outputs next to the input.

    Reads `data/<dataset>/original.csv`, renames its structural columns to the canonical names
    used throughout the codebase, extracts the temporal features, splits the log into
    train/val/test, and drops the cases longer than `data.max_seq_len_percentile` of case length
    from each split.

    The vocabularies and normalization ranges the model is built against are fit here too,
    on the train split alone, and written beside it as `dataset.json`. The declarative model is
    discovered from the same split and written to `data/<dataset>/declare/model.decl`.

    Args:
        data_config: The `data` section of this dataset's experiment config.
        declare_config: The `declare` section, driving the discovery of the declarative model.
    """
    column_mapping = {
        data_config.case_key: CASE_KEY,
        data_config.activity_key: ACTIVITY_KEY,
        data_config.resource_key: RESOURCE_KEY,
        data_config.timestamp_key: TIMESTAMP_KEY,
    }
    dataset = data_config.name

    # Read the raw log and derive the columns the model reads.
    print(f'Preprocessing "{dataset}"...')
    log = read_log(paths.original_log(dataset), column_mapping=column_mapping)
    log = preprocess(log, feature_columns=data_config.event_features)

    # Compute the cutoff in events
    max_seq_len = case_length_cutoff(log, percentile=data_config.max_seq_len_percentile)

    # Split the log into train/val/test and drop the cases longer than `max_seq_len` events.
    # Must be done in this order, otherwise the splits would be biased by the dropped cases.
    print(f'Splitting "{dataset}" into train/val/test ({data_config.split_strategy})...')
    train, val, test = (
        drop_long_cases(rows, max_seq_len=max_seq_len)
        for rows in _split(log, data_config=data_config)
    )
    cases_read = log[CASE_KEY].nunique()
    dropped = cases_read - sum(rows[CASE_KEY].nunique() for rows in (train, val, test))
    print(
        f'Dropped {dropped} of {cases_read} cases longer than {max_seq_len} events '
        f'(p{data_config.max_seq_len_percentile}, {dropped / cases_read:.2%})'
    )

    for split, rows in ((Split.TRAIN, train), (Split.VAL, val), (Split.TEST, test)):
        write_log(rows, paths.split_path(dataset=dataset, split=split))

    # Fit the vocabularies and normalization ranges on the train split, writng them out to
    # `dataset.json`. The generated values can be decoded back using the same codec.
    codec = DatasetCodec.fit(train, data_config=data_config, max_trace_length=max_seq_len)
    codec.save()

    # Discover the declarative model from the train split and write it out to `model.decl`.
    num_constraints = discover_declare_model(
        train,
        dataset=dataset,
        declare_config=declare_config,
    )

    print(
        f'Preprocessed "{dataset}": {len(train)} train, {len(val)} val, {len(test)} test '
        f'events, {len(codec.activity.vocab)} activities, '
        f'{len(codec.resource.vocab)} resources, '
        f'{len(codec.categorical_features)} categorical and '
        f'{len(codec.numeric_features)} numeric feature channels, '
        f'{num_constraints} declarative constraints'
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Turn a raw event log into the train/val/test CSVs the model consumes.'
    )
    parser.add_argument(
        '-c',
        '--config',
        type=Path,
        required=True,
        help="Path to this dataset's experiment config YAML.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run(data_config=config.data, declare_config=config.declare)


if __name__ == '__main__':
    main()
