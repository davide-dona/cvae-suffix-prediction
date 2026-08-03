from pathlib import Path
import pandas as pd

from src.logs.keys import CSV_SEPARATOR, TIMESTAMP_KEY


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
