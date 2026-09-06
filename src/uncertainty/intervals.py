from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import Axis, read_prefix_scores, require_columns, score_files
from src.uncertainty.resampling import SEED, Units, resample_means
from src.uncertainty.units import LENGTHS, OCCURRENCES, by_length

# What a band covers, and how many resamples its two ends are read off. Fewer than the 10,000
# `significance.py` draws, and deliberately: a p-value there has to survive being multiplied by the
# comparisons of a row, where a band is two order statistics of the resample means and is settled
# long before this. The draw itself is the cost here, `INTERVAL_RESAMPLES` integers per prefix of
# every run and breakdown.
LEVEL = 0.95
INTERVAL_RESAMPLES = 2_000
_TAILS = (50.0 * (1.0 - LEVEL), 50.0 * (1.0 + LEVEL))

# What `read_intervals` returns. The key columns of `read_reports` exactly, minus the two a report
# already carries, so a band joins onto the mean it is drawn around on the key.
INTERVAL_COLUMNS = ('dataset', 'model', 'axis', 'length', 'metric', 'low', 'high')


def percentile_interval(
    units: Units, *, resamples: int, generator: np.random.Generator
) -> np.ndarray:
    """Bound how far the mean of a set of units could be off, on every metric at once.

    Args:
        units: What is being bounded, from `src.uncertainty.units`.
        resamples: How many resamples to read the two ends off.
        generator: The source of the draws.
    Returns:
        The two ends of the interval, `[2, ...]`, at `_TAILS` of the resample means. Each end is
        itself a mean of the values, so a band on a metric bounded in `[0, 1]` cannot leave it.
    """
    means = list(resample_means(units, resamples, generator=generator))
    return np.percentile(np.concatenate(means, axis=0), _TAILS, axis=0)


def _rows(
    frame: pd.DataFrame, metrics: Sequence[str], axes: Sequence[Axis], identity: dict[str, str]
) -> list[dict[str, object]]:
    """Bound one run's mean at every length of every breakdown.

    Args:
        frame: That run's per-prefix scores, from `read_prefix_scores`.
        metrics: Which metrics to bound, in the order the columns of the units come back in.
        axes: Which breakdowns to bound them over.
        identity: The dataset and model every row of this run carries.
    Returns:
        One row per metric per length per breakdown, under `INTERVAL_COLUMNS`.
    """
    # Seeded per run rather than per length, so a run's bands do not depend on how many lengths
    # were asked for before them, and two runs of the pipeline still draw the same ones.
    generator = np.random.default_rng(SEED)

    rows: list[dict[str, object]] = []
    for axis in axes:
        for length, bounded, units in by_length(frame, metrics, axis=axis):
            low, high = percentile_interval(
                units, resamples=INTERVAL_RESAMPLES, generator=generator
            )
            rows.extend(
                {
                    **identity,
                    'axis': axis,
                    'length': length,
                    'metric': metric,
                    'low': float(low[column]),
                    'high': float(high[column]),
                }
                for column, metric in enumerate(bounded)
            )
    return rows


def read_intervals(
    reports: Sequence[Path],
    *,
    metrics: Sequence[str],
    axes: Sequence[Axis] = (Axis.PREFIX, Axis.SUFFIX),
) -> pd.DataFrame:
    """Bound how far each length's reported mean could be off, from the per-prefix scores beside
    each report.

    A report holds the mean of every metric at every length and says nothing about how well that
    many prefixes pin it down. That is what the per-prefix scores already hold, and this is where
    they are reduced to it: how far a mean could be off is a property of the scores it was taken
    over, not a claim a page makes, which is why it lives here rather than beside the figure that
    draws it.

    Args:
        reports: The evaluation reports being read. Each one's per-prefix scores are read from
            beside it, which is where `pipelines.evaluate` writes them.
        metrics: Which metrics of `src.evaluation.scores.METRICS` to bound. Named by the caller
            rather than taken to be all of them, a page usually drawing a handful.
        axes: Which length breakdowns to bound them over, the two `read_reports` is keyed by that a
            figure draws a line along.
    Returns:
        One row per metric per length per breakdown per run, under `INTERVAL_COLUMNS`. `length` is
        a nullable integer, as in `read_reports`, so the two frames join on the key columns.
    Raises:
        ValueError: If `Axis.OVERALL` is asked for, if a report has no per-prefix scores beside it,
            if one is not a scores file, if one log is given two runs of the same model, or if a run
            predates a metric asked for.
    """
    if Axis.OVERALL in axes:
        raise ValueError(
            'the overall breakdown cannot be bounded by an ordinary bootstrap: over the whole '
            'split a case contributes one prefix per cut point, so the prefixes are not '
            'independent and resampling them reads the interval far too narrow. Within one length '
            'a case has exactly one prefix, which is why the two length breakdowns are bounded '
            'here; a difference over the whole split is settled by the cluster bootstrap in '
            '`src.uncertainty.significance` instead.'
        )

    wanted = tuple(dict.fromkeys(metrics))
    # `OCCURRENCES` whatever was asked for: it is what says which prefixes a metric read over the
    # comparable ones is bounded on, and a file predating it predates the population itself.
    needed = tuple(dict.fromkeys((*wanted, OCCURRENCES, *(LENGTHS[axis] for axis in axes))))

    rows: list[dict[str, object]] = []
    for dataset, files in score_files(reports).items():
        for model, file in files.items():
            require_columns(file, needed)
            frame = read_prefix_scores(file, columns=needed)
            rows.extend(_rows(frame, wanted, axes, {'dataset': dataset, 'model': model}))

    intervals = pd.DataFrame(rows, columns=list(INTERVAL_COLUMNS))
    return intervals.astype({'length': 'Int64', 'low': 'float64', 'high': 'float64'})
