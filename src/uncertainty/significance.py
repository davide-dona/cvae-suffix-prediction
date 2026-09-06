from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import score_files
from src.evaluation.scores import METRICS
from src.scalar_metrics import Direction, oriented
from src.uncertainty.resampling import SEED, Units, resample_means
from src.uncertainty.units import by_case

# What a difference has to clear to be called real, and how many resamples it is read against.
# The resolution of an uncorrected p is `1 / TEST_RESAMPLES`, so this many leaves room for the
# correction to divide it by the handful of models a row compares.
ALPHA = 0.05
TEST_RESAMPLES = 10_000

# What `test_significance` returns, and what a table's emphasis is read from. One row per model per
# metric per log. `p_value` is the corrected p against that row's reference and is NaN for the
# reference itself; `best` is whether the cell is bold.
SIGNIFICANCE_COLUMNS = ('dataset', 'model', 'metric', 'p_value', 'best')

# The metrics a difference can be tested on: one with no better direction has no best value to be
# indistinguishable from, which is every property of the log and every diagnostic of a run.
TESTED = tuple(
    key for key, metric in METRICS.entries.items() if metric.direction is not Direction.NONE
)


def _directions(metrics: Sequence[str]) -> tuple[Direction, ...]:
    """Which way each of a set of metrics reads, which `oriented` turns into arithmetic.

    Args:
        metrics: The metrics being tested, in the order the units' last axis holds them.
    Returns:
        Their directions, in that same order.
    """
    return tuple(METRICS[key].direction for key in metrics)


def two_sided_p(
    units: Units,
    *,
    reference: np.ndarray,
    directions: Sequence[Direction],
    resamples: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """The share of resamples in which each model's mean falls on the wrong side of the reference.

    A paired cluster bootstrap: the unit drawn is the case, with all of its cut points attached, so
    the dependence between the prefixes of one case is carried into the test. A resample's mean for
    one model is the ratio estimator of exactly the mean the table prints, which is what
    `resample_means` reads off the units.

    Args:
        units: Each case's summed scores, `[cases, models, metrics]`, from `by_case`.
        reference: Which model each metric is compared against, `[metrics]`.
        directions: Which way each of those metrics reads, in the order the units hold them.
        resamples: How many resamples to read the p against.
        generator: The source of the draws.
    Returns:
        The uncorrected two-sided p of every model against the reference of its metric,
        `[models, metrics]`. The reference's own is 1.0, its difference from itself being 0.
    """
    _, models, metrics = units.totals.shape
    columns = np.arange(metrics)
    at_least, at_most = np.zeros((models, metrics)), np.zeros((models, metrics))

    for means in resample_means(units, resamples, generator=generator):
        # How much better than its metric's reference each model came out, [chunk, models, metrics].
        against = means[:, reference, columns][:, None, :]  # [chunk, 1, metrics]
        better = oriented(means, directions) - oriented(against, directions)
        at_least += (better >= 0).sum(axis=0)
        at_most += (better <= 0).sum(axis=0)

    # Two-sided, by inverting the bootstrap interval of the difference: a model that lands on both
    # sides of the reference is one the resamples cannot separate from it.
    return np.minimum(2.0 * np.minimum(at_least, at_most) / resamples, 1.0)


def holm(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment of one family of p-values.

    A row compares every model against one reference at once, so its p-values are one family and
    an uncorrected threshold would call a difference real once every twenty comparisons by chance.

    Args:
        p_values: The uncorrected p of each comparison, in any order.
    Returns:
        The adjusted p of each, in the order given, each capped at 1.0 and at no less than the
        adjusted p of every stricter comparison.
    """
    order = np.argsort(p_values)
    remaining = len(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (remaining - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def best_group(uncorrected: np.ndarray, *, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Which models a metric's column is emphasized on, and what each one's corrected p is.

    The one place the best group is defined: the reference plus everything `ALPHA` cannot separate
    from it, over the comparisons of one row corrected together.

    Args:
        uncorrected: The two-sided p of every model against the reference of its metric,
            `[models, metrics]`, from `two_sided_p`.
        reference: Which model each metric is compared against, `[metrics]`.
    Returns:
        The corrected p of every model, `[models, metrics]` and NaN on each metric's own reference,
        and whether each is emphasized, `[models, metrics]`.
    """
    models, metrics = uncorrected.shape
    corrected = np.full((models, metrics), np.nan)
    best = np.zeros((models, metrics), dtype=bool)

    for column in range(metrics):
        others = [index for index in range(models) if index != reference[column]]
        adjusted = holm(uncorrected[others, column])
        corrected[others, column] = adjusted
        best[others, column] = adjusted >= ALPHA
        best[reference[column], column] = True
    return corrected, best


def test_significance(reports: Sequence[Path]) -> pd.DataFrame:
    """Tell, for every metric of every log, which models are indistinguishable from the best.

    The best value alone says a model won where the honest claim is usually that several of them
    are tied. Each log's models are compared over the prefixes all of them scored, by a paired
    bootstrap that resamples whole cases, and the best group is the best model plus everything the
    resamples cannot separate from it.

    Args:
        reports: The evaluation reports being tabulated. Each one's per-prefix scores are read from
            beside it, which is where `pipelines.evaluate` writes them.
    Returns:
        One row per model per metric per log, under `SIGNIFICANCE_COLUMNS`. Only metrics with a
        declared direction are tested, since one with no better value has no best to be tied with.
        A log holding a single model marks nothing, as a row with a single column does.
    Raises:
        ValueError: If a report has no per-prefix scores beside it, if one log is given two runs of
            the same model, or if the models of a log do not score the same prefixes.
    """
    rows: list[dict[str, object]] = []
    for dataset, files in score_files(reports).items():
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
                for key in TESTED
            )
            continue

        # Once per population: a metric read over the comparable prefixes alone is tested over
        # the cases those fall in, so a p is read against the mean the table prints.
        for metrics, units in by_case(dataset=dataset, files=files, metrics=TESTED):
            directions = _directions(metrics)
            # The mean the table prints: every prefix weighing the same, so a long case weighs more.
            reference = oriented(units.mean, directions).argmax(axis=0)  # [metrics]

            generator = np.random.default_rng(SEED)
            uncorrected = two_sided_p(
                units,
                reference=reference,
                directions=directions,
                resamples=TEST_RESAMPLES,
                generator=generator,
            )
            corrected, best = best_group(uncorrected, reference=reference)

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
