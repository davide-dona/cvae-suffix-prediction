import argparse

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src import paths
from src.cli import banner, step
from src.configs import DataConfig, DeclareConfig, load_dataset_config
from src.datasets.codec import DatasetCodec
from src.logs import (
    CASE_ELAPSED_KEY,
    CASE_KEY,
    CYCLE_TIME_KEY,
    DAY_COS_KEY,
    DAY_SIN_KEY,
    MIN_PREFIX_KEY,
    MISSING_FEATURE,
    REMAINING_TIME_KEY,
    RESOURCE_KEY,
    SECONDS_COS_KEY,
    SECONDS_SIN_KEY,
    TIMESTAMP_KEY,
    ContinuationIndex,
    Split,
    read_original_log,
    write_log,
)
from src.logs.declare.discovery import discover_declare_model
from src.logs.preprocessing import (
    add_calendar,
    add_case_elapsed,
    add_cycle_time,
    add_remaining_time,
    case_durations,
    drop_cases_by_duration,
    drop_cases_by_length,
    out_of_time_split,
    sort_log,
)


def case_length_cutoff(log: pd.DataFrame, *, data_config: DataConfig) -> int:
    """Find the cutoff in events for dropping cases too long to fit the model's sequence tensors.

    Args:
        log: The whole preprocessed log, one row per event, before it is split.
        data_config: The `data` section, which either states the cutoff outright or names the
            percentile of case length to read it off at.
    Returns:
        The cutoff in events.
    """
    lengths = log.groupby(CASE_KEY)[CASE_KEY].size()
    return int(np.ceil(np.percentile(lengths, data_config.max_seq_len_percentile)))


def case_duration_cutoff(log: pd.DataFrame, *, data_config: DataConfig) -> float:
    """Find the cutoff in days for dropping the cases whose duration is not a real one.

    Args:
        log: The whole log as read, one row per event, before it is split.
        data_config: The `data` section, which names the percentile of case duration to read the
            cutoff off at.
    Returns:
        The cutoff in days.
    """
    durations = case_durations(log, case_key=CASE_KEY, timestamp_key=TIMESTAMP_KEY)
    return float(np.percentile(durations, data_config.max_case_duration_percentile))


def report_dropped(*, before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Print how many cases a filter dropped, under the step that ran it.

    Args:
        before: The log the filter was handed.
        after: What it gave back.
    """
    cases_read = before[CASE_KEY].nunique()
    dropped = cases_read - after[CASE_KEY].nunique()
    print(
        f'  dropped {dropped:,} of {cases_read:,} cases ({dropped / cases_read:.2%})',
        flush=True,
    )


def add_prefix_bounds(log: pd.DataFrame) -> pd.DataFrame:
    """Add the lower bound of the cut points every case may be split at, at its full width.
    Initialized to 1, updated to the actual lower bound after the out-of-time split.
    Args:
        log: The preprocessed log, one row per event.
    Returns:
        A copy of `log` with `MIN_PREFIX_KEY` added, constant over each case's rows.
    """
    log = log.copy()
    log[MIN_PREFIX_KEY] = 1
    return log


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
        A copy of `log` with the two timestamp proxies, the remaining time and the four calendar
        columns added, and its categorical columns filled.
    """
    log = add_cycle_time(
        log,
        case_key=CASE_KEY,
        timestamp_key=TIMESTAMP_KEY,
        out_key=CYCLE_TIME_KEY,
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
        day_sin_key=DAY_SIN_KEY,
        day_cos_key=DAY_COS_KEY,
        seconds_sin_key=SECONDS_SIN_KEY,
        seconds_cos_key=SECONDS_COS_KEY,
    )
    log[RESOURCE_KEY] = log[RESOURCE_KEY].fillna(MISSING_FEATURE).astype(str)

    for column in feature_columns:
        if not is_numeric_dtype(log[column]):
            log[column] = log[column].fillna(MISSING_FEATURE).astype(str)
    return log


def run(data_config: DataConfig, declare_config: DeclareConfig) -> None:
    """
    Preprocess and split a dataset, writing outputs next to the input.

    Reads `data/<dataset>/original.csv`, renames its structural columns to the canonical names
    used throughout the codebase, drops the cases whose duration is not a real one and the cases
    too long to fit the model's sequence tensors, sorts it, extracts the temporal features, and
    splits what is left into train/val/test out of time.

    The vocabularies and normalization statistics the model is built against are fit here too,
    on the train split alone, and written beside it as `dataset.json`.

    The continuations each held-out split takes after each of its prefixes are indexed next, one
    index per split, beside the splits: training selects checkpoints against the validation
    split's and evaluation scores against the test split's, so both are always built. The
    declarative model discovered from the train split follows, and is what evaluation checks
    conformance against.

    Args:
        data_config: The `data` section of this dataset's experiment config.
        declare_config: The `declare` section, driving the discovery of the declarative model.
    """
    dataset = data_config.name

    banner(
        f'Preprocessing "{dataset}"',
        {
            'original log': paths.ORIGINAL_LOG.path(dataset),
            'split': f'{data_config.train_split:.0%} train, {data_config.val_split:.0%} val, '
            f'{data_config.test_split:.0%} test, out of time',
            'splits': paths.PROCESSED_SPLIT.directory(dataset),
            'codec': paths.CODEC.path(dataset),
            'continuations': paths.CONTINUATIONS.directory(dataset),
            'declarative model': paths.DECLARE_MODEL.path(dataset),
        },
    )

    # Read the raw log.
    with step('Reading the original log'):
        log = read_original_log(data_config)
        print(f'  {len(log):,} events over {log[CASE_KEY].nunique():,} cases', flush=True)

    # Both cutoffs are read off the log as it came, so neither filter's threshold depends on the
    # other having run. Both are applied before the split.
    max_duration_days = case_duration_cutoff(log, data_config=data_config)
    max_seq_len = case_length_cutoff(log, data_config=data_config)

    with step(f'Dropping the cases running longer than {max_duration_days:,.1f} days'):
        kept = drop_cases_by_duration(
            log,
            case_key=CASE_KEY,
            timestamp_key=TIMESTAMP_KEY,
            max_duration_days=max_duration_days,
        )
        report_dropped(before=log, after=kept)
        log = kept

    with step(f'Dropping the cases longer than {max_seq_len} events'):
        kept = drop_cases_by_length(log, case_key=CASE_KEY, max_seq_len=max_seq_len)
        report_dropped(before=log, after=kept)
        log = kept

    # Derive the columns the model reads, all of which are read off neighbouring rows or off the
    # size of a case, so the log has to be in order first.
    with step('Sorting the log and deriving the temporal and calendar features'):
        log = sort_log(log, case_key=CASE_KEY, timestamp_key=TIMESTAMP_KEY)
        log = preprocess(log, feature_columns=data_config.event_features)
        log = add_prefix_bounds(log)

    with step('Splitting the log out of time'):
        train, val, test = out_of_time_split(
            log,
            case_key=CASE_KEY,
            timestamp_key=TIMESTAMP_KEY,
            val_frac=data_config.val_split,
            test_frac=data_config.test_split,
        )

    with step('Writing the splits'):
        for split, rows in ((Split.TRAIN, train), (Split.VAL, val), (Split.TEST, test)):
            write_log(rows, paths.PROCESSED_SPLIT.path(dataset=dataset, split=split))

    # Fit the vocabularies and normalization statistics on the train split, writng them out to
    # `dataset.json`. The generated values can be decoded back using the same codec.
    with step('Fitting the dataset codec on the train split'):
        codec = DatasetCodec.fit(train, data_config=data_config, max_trace_length=max_seq_len)
        codec.save()

    # Both held-out splits, since training selects on the validation split's continuations and
    # evaluation scores against the test split's.
    indexed = {}
    for split, data in ((Split.VAL, val), (Split.TEST, test)):
        with step(f'Indexing the continuations of the {split} split'):
            index = ContinuationIndex.of(
                data,
                vocabulary=codec.activity.vocab,
                names=codec.activity.names,
            )
            index.write(dataset=dataset, split=split)
            indexed[split] = index.prefixes
            print(
                f'  {index.occurrences:,} cut points over {index.prefixes:,} distinct prefixes',
                flush=True,
            )

    with step('Discovering the declarative model'):
        constraints = discover_declare_model(
            train,
            dataset=dataset,
            declare_config=declare_config,
        )
    declare_summary = f'{constraints} declarative constraints'

    print(
        f'Preprocessed "{dataset}": {len(train):,} train, {len(val):,} val, {len(test):,} test '
        f'events, {len(codec.activity.vocab)} activities, '
        f'{len(codec.resource.vocab)} resources, '
        f'{len(codec.categorical_features)} categorical and '
        f'{len(codec.numeric_features)} numeric feature channels, '
        f'{indexed[Split.VAL]:,} val and {indexed[Split.TEST]:,} test indexed prefixes, '
        f'{declare_summary}',
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Turn a raw event log into the train/val/test CSVs the model consumes.'
    )
    parser.add_argument(
        '-c',
        '--config',
        type=paths.existing_file,
        metavar='CONFIG',
        required=True,
        help="Path to this experiment's dataset config, e.g. config/datasets/bpic17.yaml.",
    )
    args = parser.parse_args()

    config = load_dataset_config(args.config)

    run(data_config=config.data, declare_config=config.declare)


if __name__ == '__main__':
    main()
