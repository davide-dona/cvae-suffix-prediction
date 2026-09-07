from collections.abc import Iterator

import numpy as np

BATCH_SIZE = 250


def resample_means(
    totals: np.ndarray,
    counts: np.ndarray,
    *,
    resamples: int,
    generator: np.random.Generator,
) -> Iterator[np.ndarray]:
    """Yield paired case-bootstrap means in batches of at most `BATCH_SIZE`.

    Args:
        totals: Summed prefix scores per case, `[cases, ...]`.
        counts: Eligible prefix counts per case, `[cases]`, all positive.
        resamples: Positive number of bootstrap draws.
        generator: Random source shared by all models and metrics in each draw.

    Yields:
        `[batch, ...]` means, with every eligible prefix weighted equally.
    """
    cases = len(counts)
    flat = totals.reshape(cases, -1)
    for start in range(0, resamples, BATCH_SIZE):
        size = min(BATCH_SIZE, resamples - start)
        picks = generator.integers(low=0, high=cases, size=(size, cases))
        picks += (np.arange(size) * cases)[:, None]
        multiplicities = np.bincount(picks.ravel(), minlength=size * cases).reshape(size, cases)
        means = (multiplicities @ flat) / (multiplicities @ counts)[:, None]
        yield means.reshape(size, *totals.shape[1:])
