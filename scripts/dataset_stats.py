from __future__ import annotations

import hydra
import pandas as pd
from omegaconf import DictConfig

from src import paths
from src.cli import banner
from src.logs.io import read_log
from src.logs.keys import (
    ACTIVITY_KEY,
    CASE_KEY,
    DAY_COS_KEY,
    DAY_SIN_KEY,
    MIN_PREFIX_KEY,
    REMAINING_TIME_KEY,
    SECONDS_COS_KEY,
    SECONDS_SIN_KEY,
    TIMESTAMP_KEY,
    Split,
)
from src.logs.preprocessing.cases import case_durations
from src.runtime import start_stage

# Columns preprocessing derives from the timestamp rather than the raw dataset carrying them.
_DERIVED_COLUMNS = {
    REMAINING_TIME_KEY,
    MIN_PREFIX_KEY,
    DAY_SIN_KEY,
    DAY_COS_KEY,
    SECONDS_SIN_KEY,
    SECONDS_COS_KEY,
}


def read_processed_log(dataset: str) -> pd.DataFrame:
    """Read the full processed log a dataset's splits partition, before the out-of-time split.

    Concatenates the three splits rather than rereading and reprocessing the raw log: splitting
    only partitions cases by time and narrows `MIN_PREFIX_KEY` on the ones crossing the
    boundary, so no row is added or dropped by it.

    Args:
        dataset: The dataset whose splits to read, from where preprocessing wrote them.
    Returns:
        The processed log, one row per event.
    """
    splits = [
        read_log(paths.PROCESSED_SPLIT.require(dataset=dataset, split=split)) for split in Split
    ]
    return pd.concat(splits, ignore_index=True)


def is_case_level(log: pd.DataFrame, column: str) -> bool:
    """Whether a feature is constant within every case rather than varying event to event."""
    return log.groupby(CASE_KEY)[column].nunique(dropna=False).le(1).all()


def summarize_dataset(dataset: str, *, event_features: list[str]) -> None:
    """Print the size and feature counts of one dataset's processed log.

    Args:
        dataset: The dataset to summarize.
        event_features: `data.event_features` from its config.
    """
    log = read_processed_log(dataset)
    lengths = log.groupby(CASE_KEY).size()
    durations = case_durations(log, case_key=CASE_KEY, timestamp_key=TIMESTAMP_KEY)
    variants = log.groupby(CASE_KEY)[ACTIVITY_KEY].agg(tuple).nunique()

    features = [column for column in event_features if column not in _DERIVED_COLUMNS]
    case_features = sum(is_case_level(log, column) for column in features)

    banner(
        f'"{dataset}"',
        {
            'cases': f'{lengths.size:,}',
            'events': f'{len(log):,}',
            'variants': f'{variants:,}',
            'activities': f'{log[ACTIVITY_KEY].nunique():,}',
            'case length': f'{lengths.mean():.1f} +/- {lengths.std():.1f} events',
            'case duration': f'{durations.mean():.2f} +/- {durations.std():.2f} days',
            'case features': case_features,
            'event features': len(features) - case_features,
        },
    )


@hydra.main(version_base='1.3', config_path='../config', config_name='dataset_stats')
def main(cfg: DictConfig) -> None:
    start_stage(cfg)
    summarize_dataset(cfg.data.name, event_features=list(cfg.data.event_features))


if __name__ == '__main__':
    main()
