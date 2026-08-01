import pandas as pd


def add_case_offset(
    log: pd.DataFrame,
    *,
    case_key: str,
    timestamp_key: str,
    out_key: str,
) -> pd.DataFrame:
    """
    Add the case's offset from the log's start, in minutes.
    Every event in a case gets the same value: the number of minutes between
    the first event of the whole log and the first event of that case.
    Args:
        log: Event log, one row per event.
        case_key: Column identifying the case each event belongs to.
        timestamp_key: Column holding the (already parsed) event timestamp.
        out_key: Name of the column to add.
    Returns:
        A copy of `log` with `out_key` added.
    """
    log = log.copy()

    log_start = log[timestamp_key].min()
    # Get the timestamp of the first event in each case
    case_start = log.groupby(case_key)[timestamp_key].transform('min')
    # Compute the offset in minutes and add it as a new column
    log[out_key] = (case_start - log_start).dt.total_seconds() / 60.0

    return log


def add_remaining_time(
    log: pd.DataFrame,
    *,
    case_key: str,
    timestamp_key: str,
    out_key: str,
) -> pd.DataFrame:
    """
    Add the number of minutes from each event to the last event of its case.
    Args:
        log: Event log, one row per event.
        case_key: Column identifying the case each event belongs to.
        timestamp_key: Column holding the (already parsed) event timestamp.
        out_key: Name of the column to add.
    Returns:
        A copy of `log` with `out_key` added, 0.0 at the last event of every case.
    """
    log = log.copy()

    # The case's last timestamp, broadcast back over its events.
    case_end = log.groupby(case_key)[timestamp_key].transform('max')
    log[out_key] = (case_end - log[timestamp_key]).dt.total_seconds() / 60.0

    return log


def add_event_delta(
    log: pd.DataFrame,
    *,
    case_key: str,
    timestamp_key: str,
    out_key: str,
) -> pd.DataFrame:
    """
    Add the number of minutes since the previous event in the same case.
    Args:
        log: Event log, one row per event.
        case_key: Column identifying the case each event belongs to.
        timestamp_key: Column holding the (already parsed) event timestamp.
        out_key: Name of the column to add.
    Returns:
        A copy of `log` with `out_key` added.
    """
    log = log.copy()
    # Compute the time difference between consecutive events in the same case
    delta = log.groupby(case_key)[timestamp_key].diff()
    # Fill NaN values (first event in each case) with 0 and convert to minutes
    log[out_key] = (delta.dt.total_seconds() / 60.0).fillna(0.0)

    return log
