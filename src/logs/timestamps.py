import pandas as pd


def add_case_elapsed(
    log: pd.DataFrame,
    *,
    case_key: str,
    timestamp_key: str,
    out_key: str,
) -> pd.DataFrame:
    """
    Add the number of minutes from the first event of a case to each of its events.
    Args:
        log: Event log, one row per event.
        case_key: Column identifying the case each event belongs to.
        timestamp_key: Column holding the (already parsed) event timestamp.
        out_key: Name of the column to add.
    Returns:
        A copy of `log` with `out_key` added, 0.0 at the first event of every case.
    """
    log = log.copy()

    # The case's first timestamp, broadcast back over its events.
    case_start = log.groupby(case_key)[timestamp_key].transform('min')
    log[out_key] = (log[timestamp_key] - case_start).dt.total_seconds() / 60.0

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


def add_calendar(
    log: pd.DataFrame,
    *,
    timestamp_key: str,
    day_key: str,
    seconds_key: str,
) -> pd.DataFrame:
    """
    Add where in the week and where in the day each event happened.

    Both are read off the event's own timestamp rather than off its case, so neither says
    anything about how far along the case is; what they carry is the working rhythm behind it.

    Args:
        log: Event log, one row per event.
        timestamp_key: Column holding the (already parsed) event timestamp.
        day_key: Name of the column to add for the day of the week, 0 (Monday) to 6 (Sunday).
        seconds_key: Name of the column to add for the seconds since midnight, 0 to 86399.
    Returns:
        A copy of `log` with both columns added.
    """
    log = log.copy()

    timestamps = log[timestamp_key].dt
    log[day_key] = timestamps.weekday
    log[seconds_key] = timestamps.hour * 3600 + timestamps.minute * 60 + timestamps.second

    return log
