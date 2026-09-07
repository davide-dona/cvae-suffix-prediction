from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import score_files
from src.evaluation.scores import METRICS
from src.scalar_metrics import Direction, oriented
from src.uncertainty.resampling import resample_means
from src.uncertainty.units import by_case

ALPHA = 0.05
RESAMPLES = 10_000
SEED = 42
SIGNIFICANCE_COLUMNS = ('dataset', 'model', 'metric', 'p_value', 'best')
RANKED = tuple(
    key for key, metric in METRICS.entries.items() if metric.direction is not Direction.NONE
)


def two_sided_p(
    totals: np.ndarray,
    counts: np.ndarray,
    *,
    resamples: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Estimate null-centered two-sided bootstrap p-values for paired differences.

    Args:
        totals: Case sums of paired prefix differences, `[cases, pairs, metrics]`.
        counts: Eligible prefix counts per case, `[cases]`.
        resamples: Positive number of bootstrap draws.
        generator: Random source for paired case resampling.

    Returns:
        Approximate p-values, `[pairs, metrics]`. Fewer than two cases or a constant
        bootstrap distribution give NaN. Independent, representative cases and a
        sufficiently large case sample are required; these are not exact null tests.

    Raises:
        ValueError: If the requested number of resamples is not positive.
    """
    if resamples < 1:
        raise ValueError('resamples must be positive.')
    if len(counts) < 2:
        return np.full(totals.shape[1:], np.nan)

    observed = totals.sum(axis=0) / counts.sum()
    extreme = np.zeros(observed.shape, dtype=np.int64)
    smallest = np.full(observed.shape, np.inf)
    largest = np.full(observed.shape, -np.inf)
    # Roundoff at the scale of case differences must not separate equal tail values.
    scale = np.max(np.abs(totals / counts[:, None, None]), axis=0)
    tolerance = 100 * np.finfo(np.float64).eps * scale
    for differences in resample_means(
        totals=totals, counts=counts, resamples=resamples, generator=generator
    ):
        extreme += (np.abs(differences - observed) >= np.abs(observed) - tolerance).sum(axis=0)
        smallest = np.minimum(smallest, differences.min(axis=0))
        largest = np.maximum(largest, differences.max(axis=0))

    p_values = (extreme + 1) / (resamples + 1)
    p_values[largest - smallest <= tolerance] = np.nan
    return p_values


def holm(p_values: np.ndarray) -> np.ndarray:
    """Adjust one family of p-values, retaining unavailable comparisons in its size.

    NaN comparisons count as p=1 for correction and remain NaN in the result.
    """
    available = np.isfinite(p_values)
    values = np.where(available, p_values, 1.0)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(
        np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)), 1.0
    )
    return np.where(available, adjusted, np.nan)


def compare_means(
    totals: np.ndarray,
    counts: np.ndarray,
    *,
    directions: Sequence[Direction],
    resamples: int,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return adjusted p-values against the observed best and table emphasis.

    Args:
        totals: Case score sums, `[cases, models, metrics]`, in stable model order.
        counts: Eligible prefix counts per case, `[cases]`.
        directions: Ranking direction of each metric; ZERO is descriptive only.
        resamples: Number of paired bootstrap draws.
        generator: Random source for case resampling.

    Returns:
        Two `[models, metrics]` arrays: adjusted p-values and emphasis flags. Each
        dataset/metric's correction family contains every model pair. References,
        calibration metrics, and unavailable tests have NaN p-values. Emphasis
        means observed best or no detected difference, not evidence of equivalence.
    """
    means = totals.sum(axis=0) / counts.sum()
    scores = oriented(values=means, directions=directions)
    reference = scores.argmax(axis=0)
    best = scores == scores.max(axis=0)
    p_values = np.full(means.shape, np.nan)
    tested = [
        i
        for i, direction in enumerate(directions)
        if direction in (Direction.HIGHER, Direction.LOWER)
    ]
    if not tested:
        return p_values, best

    left, right = np.triu_indices(len(means), k=1)
    differences = totals[:, left][:, :, tested] - totals[:, right][:, :, tested]
    uncorrected = two_sided_p(
        totals=differences, counts=counts, resamples=resamples, generator=generator
    )
    for column, metric in enumerate(tested):
        adjusted = holm(uncorrected[:, column])
        for pair, (first, second) in enumerate(zip(left, right, strict=True)):
            if first == reference[metric]:
                p_values[second, metric] = adjusted[pair]
            elif second == reference[metric]:
                p_values[first, metric] = adjusted[pair]
    best |= p_values >= ALPHA
    return p_values, best


def test_significance(reports: Sequence[Path]) -> pd.DataFrame:
    """Compare reported means with a paired case bootstrap and all-pairs Holm correction.

    Calibration emphasis is descriptive: smallest absolute mean gap. Single-model
    datasets have no emphasis. Populations with no eligible prefixes are omitted.

    Args:
        reports: Evaluation reports with per-prefix score files beside them.

    Returns:
        A dataframe under `SIGNIFICANCE_COLUMNS`. `p_value` is the adjusted p-value
        against the observed best, or NaN when unavailable. `best` controls emphasis.

    Raises:
        ValueError: If scores are missing, model runs are duplicated, prefixes or
            occurrence counts differ, or eligible scores are nonfinite.
    """
    rows: list[dict[str, object]] = []
    for dataset, files in sorted(score_files(reports).items()):
        files = dict(sorted(files.items()))
        models = list(files)
        if len(models) < 2:
            rows.extend(
                {
                    'dataset': dataset,
                    'model': models[0],
                    'metric': key,
                    'p_value': np.nan,
                    'best': False,
                }
                for key in RANKED
            )
            continue

        for metrics, totals, counts in by_case(dataset=dataset, files=files, metrics=RANKED):
            corrected, best = compare_means(
                totals=totals,
                counts=counts,
                directions=[METRICS[key].direction for key in metrics],
                resamples=RESAMPLES,
                generator=np.random.default_rng(SEED),
            )
            rows.extend(
                {
                    'dataset': dataset,
                    'model': model,
                    'metric': key,
                    'p_value': float(corrected[index, column]),
                    'best': bool(best[index, column]),
                }
                for column, key in enumerate(metrics)
                for index, model in enumerate(models)
            )

    frame = pd.DataFrame(rows, columns=list(SIGNIFICANCE_COLUMNS))
    return frame.astype({'p_value': 'float64', 'best': 'bool'})
