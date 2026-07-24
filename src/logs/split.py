import pandas as pd


def temporal_split(
    log: pd.DataFrame,
    *,
    case_key: str,
    timestamp_key: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a log into train/val/test by case start time.
    
    Cases are sorted by the timestamp of their first event, then cut into three
    contiguous blocks so validation and test cases always start later than the
    training cases they are evaluated against.

    Args:
        log: Event log, one row per event.
        case_key: Column identifying the case each event belongs to.
        timestamp_key: Column holding the (already parsed) event timestamp.
        train_frac: Fraction of cases assigned to the training set.
        val_frac: Fraction of cases assigned to the validation set.
        test_frac: Fraction of cases assigned to the test set.
    Returns:
        `(train, val, test)` DataFrames, each a row-subset of `log`.
    Raises:
        ValueError: If the fractions do not sum to 1.
    """
    # Check that splits sum to 1.0
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f'train/val/test fractions must sum to 1.0, got {total}')

    # Get the first event timestamp for each case and sort cases by that timestamp
    case_start = log.groupby(case_key)[timestamp_key].min().sort_values()
    cases = case_start.index.to_list()
    
    # Compute the number of cases for each split
    n = len(cases)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    # Split the cases into train/val/test based on the computed indices
    train_cases = cases[:n_train]
    val_cases = cases[n_train:n_train + n_val]
    test_cases = cases[n_train + n_val:]
    assert len(train_cases) + len(val_cases) + len(test_cases) == n
    
    train = log[log[case_key].isin(train_cases)]
    val = log[log[case_key].isin(val_cases)]
    test = log[log[case_key].isin(test_cases)]

    return train, val, test
