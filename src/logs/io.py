from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src import paths
from src.logs.keys import (
    ACTIVITY_KEY,
    CASE_KEY,
    CSV_SEPARATOR,
    RESOURCE_KEY,
    TIMESTAMP_KEY,
)

if TYPE_CHECKING:
    from src.configs import DataConfig


def read_log(
    path: str | Path,
    *,
    separator: str = CSV_SEPARATOR,
    column_mapping: dict[str, str] | None = None,
    timestamp_key: str = TIMESTAMP_KEY,
    dtype: dict[str, type] | None = None,
) -> pd.DataFrame:
    """
    Read a CSV event log into a DataFrame, optionally renaming columns and parsing timestamps.
    Args:
        path: Path to the CSV file.
        separator: Field separator used by the CSV.
        column_mapping: Optional `{raw_name: canonical_name}` renaming applied
            right after reading. Only column names are touched; values are
            never transformed.
        timestamp_key: Column (after renaming) holding the event timestamp,
            parsed into `datetime64`.
        dtype: Optional `{raw_name: dtype}` forced on those columns instead of letting
            pandas infer them. Named before the renaming, like every `pd.read_csv` argument.

    Returns:
        The log as a DataFrame, one row per event.
    """
    path = Path(path)
    log = pd.read_csv(path, sep=separator, dtype=dtype)

    if column_mapping:
        log = log.rename(columns=column_mapping)

    log[timestamp_key] = pd.to_datetime(log[timestamp_key], format='mixed')

    return log


def read_original_log(data_config: DataConfig) -> pd.DataFrame:
    """Read a dataset's raw log, renaming its structural columns to the canonical ones.

    Args:
        data_config: The `data` section naming the dataset, the raw log's separator and the
            raw columns holding the case, the activity, the resource and the timestamp.

    Returns:
        The raw log as a DataFrame, one row per event, with no derived column added yet.
    """
    return read_log(
        paths.ORIGINAL_LOG.path(data_config.name),
        separator=data_config.separator,
        column_mapping={
            data_config.case_key: CASE_KEY,
            data_config.activity_key: ACTIVITY_KEY,
            data_config.resource_key: RESOURCE_KEY,
            data_config.timestamp_key: TIMESTAMP_KEY,
        },
        # The case identifier is a name, never a number: a log that numbers its cases would
        # otherwise be read as integers, which is a dtype pm4py refuses when the declarative
        # model is discovered and which every reader downstream already overrides to text.
        dtype=dict.fromkeys([data_config.case_key, *data_config.string_features], str),
    )


def write_log(
    log: pd.DataFrame,
    path: str | Path,
    *,
    separator: str = CSV_SEPARATOR,
    index: bool = False,
) -> Path:
    """Write an event log to CSV.

    Args:
        log: The log to write.
        path: Destination path. Parent directories are created if missing.
        separator: Field separator to use.
        index: Whether to write the DataFrame index as a column.

    Returns:
        The path the log was written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(path, sep=separator, index=index)
    return path
