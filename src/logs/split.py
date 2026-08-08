import numpy as np
import pandas as pd

# The seed U-ED-LSTM's loader notebooks set before building their splitter. Fixed here rather
# than configurable.
UEDLSTM_SEED = 17


def temporal_split(
    log: pd.DataFrame, *, case_key: str, timestamp_key: str, train_frac: float, val_frac: float
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
    Returns:
        `(train, val, test)` DataFrames, each a row-subset of `log`.
    """
    # Get the first event timestamp for each case and sort cases by that timestamp
    case_start = log.groupby(case_key)[timestamp_key].min().sort_values()
    cases = case_start.index.to_list()

    # Compute the number of cases for each split
    n = len(cases)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    # Split the cases into train/val/test based on the computed indices
    train_cases = cases[:n_train]
    val_cases = cases[n_train : n_train + n_val]
    test_cases = cases[
        n_train + n_val :
    ]  # Assumed to be the remainder, so no rounding issues arise
    assert len(train_cases) + len(val_cases) + len(test_cases) == n

    train = log[log[case_key].isin(train_cases)]
    val = log[log[case_key].isin(val_cases)]
    test = log[log[case_key].isin(test_cases)]

    return train, val, test


def uedlstm_split(
    log: pd.DataFrame, *, case_key: str, val_frac: float, test_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a log the way U-ED-LSTM does, so both models can be scored on identical test cases.

    A seeded shuffle of the sorted case list, cut into a validation block, a test block and the
    remainder.

    Random rather than chronological, and with no leakage prevention.
    Equivalent to `EventLogSplitter.split`, which is the same code in U-ED-LSTM's
    `new_event_log_loader.py` and `new_event_log_loader_v2.py`; only the fractions each
    dataset passes it differ.

    Args:
        log: Event log, one row per event.
        case_key: Column identifying the case each event belongs to.
        val_frac: Fraction of cases assigned to the validation set, taken first.
        test_frac: Fraction of cases assigned to the test set, taken next.
    Returns:
        `(train, val, test)` DataFrames, each a row-subset of `log`. A case whose ID is missing
        appears in none of them.
    """
    # Sorted, and with the missing IDs dropped, because that is the case list their splitter
    # receives: it shuffles what their `groupby` hands back, which drops the group whose key
    # read as NaN - Sepsis has a case named "NA" - and which came back sorted under the pandas
    # their notebooks were run with. Both the order and the count feed the permutation, so a
    # deviation in either gives a different split from the same seed.
    # The object array is theirs too: numpy warns that shuffling a pandas extension array in
    # place is not guaranteed to be a permutation of it.
    cases = np.asarray(sorted(log[case_key].dropna().unique()), dtype=object)
    # The legacy MT19937 generator, which is what `np.random.seed` + `np.random.shuffle` draw
    # from. `default_rng` would be a different generator, so a different permutation.
    np.random.RandomState(seed=UEDLSTM_SEED).shuffle(cases)

    # Their splitter floors each block's size independently and leaves train the remainder.
    n = len(cases)
    n_val = int(val_frac * n)
    n_test = int(test_frac * n)

    val_cases = cases[:n_val]
    test_cases = cases[n_val : n_val + n_test]
    train_cases = cases[n_val + n_test :]

    train = log[log[case_key].isin(train_cases)]
    val = log[log[case_key].isin(val_cases)]
    test = log[log[case_key].isin(test_cases)]

    return train, val, test
