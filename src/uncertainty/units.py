from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import read_prefix_scores, require_columns
from src.evaluation.scores import COMPARABLE_METRICS, MIN_REFERENCE_OCCURRENCES

OCCURRENCES = 'reference_occurrences'
PREFIX_KEYS = ('case_id', 'prefix_len')


def _aligned(
    dataset: str, files: dict[str, Path], metrics: Sequence[str]
) -> tuple[np.ndarray, pd.MultiIndex, np.ndarray]:
    """Align model scores on identical sorted prefix keys and occurrence counts."""
    frames = []
    prefixes = None
    occurrences = None
    for file in files.values():
        columns = (*PREFIX_KEYS, *metrics, OCCURRENCES)
        require_columns(path=file, columns=columns)
        frame = read_prefix_scores(path=file, columns=columns)
        if frame[list(PREFIX_KEYS)].isna().any().any():
            raise ValueError(f'{file} contains missing prefix keys.')
        frame = frame.set_index(list(PREFIX_KEYS)).sort_index()
        if frame.index.has_duplicates:
            raise ValueError(f'{file} scores the same prefix twice, so it cannot be compared.')
        if prefixes is not None and not frame.index.equals(prefixes):
            raise ValueError(f'The runs of {dataset} do not score the same prefixes.')
        prefixes = frame.index
        current = frame[OCCURRENCES].to_numpy(dtype=np.float64)
        if not np.isfinite(current).all() or (current < 0).any():
            raise ValueError(f'{file} contains invalid reference occurrence counts.')
        if occurrences is not None and not np.array_equal(current, occurrences):
            raise ValueError(f'The runs of {dataset} have different reference occurrence counts.')
        occurrences = current
        frames.append(frame[list(metrics)].to_numpy(dtype=np.float64))

    assert prefixes is not None and occurrences is not None
    return np.stack(frames, axis=1), prefixes, occurrences


def by_case(
    dataset: str, files: dict[str, Path], metrics: Sequence[str]
) -> Iterator[tuple[list[str], np.ndarray, np.ndarray]]:
    """Yield case totals and prefix counts for each reporting population.

    Args:
        dataset: Dataset name used in validation errors.
        files: Per-prefix score files in model order.
        metrics: Metrics to aggregate, in output order.

    Yields:
        Metric names, score totals `[cases, models, metrics]`, and eligible prefix
        counts `[cases]`. Distributional metrics use comparable prefixes; other
        metrics use every prefix. Empty populations are omitted.

    Raises:
        ValueError: If required columns are missing, prefix keys are invalid or
            differ between models, occurrence counts differ or are invalid, or
            eligible scores are nonfinite.
    """
    values, prefixes, occurrences = _aligned(dataset=dataset, files=files, metrics=metrics)
    groups: dict[bool, list[int]] = {}
    for column, metric in enumerate(metrics):
        groups.setdefault(metric in COMPARABLE_METRICS, []).append(column)

    for comparable, columns in groups.items():
        keep = (
            occurrences >= MIN_REFERENCE_OCCURRENCES
            if comparable
            else np.ones(len(occurrences), dtype=bool)
        )
        if not keep.any():
            continue
        group = [metrics[column] for column in columns]
        part = values[keep][:, :, columns]
        if not np.isfinite(part).all():
            raise ValueError(f'{dataset} has nonfinite eligible scores for {", ".join(group)}.')
        # Sorted prefix keys keep each case's eligible cuts contiguous.
        cases = prefixes[keep].get_level_values('case_id').to_numpy()
        starts = np.flatnonzero(np.r_[True, cases[1:] != cases[:-1]])
        totals = np.add.reduceat(part, starts, axis=0)
        counts = np.diff(np.r_[starts, len(cases)])
        yield group, totals, counts
