from collections.abc import Iterator
from dataclasses import dataclass
from typing import Self

import numpy as np

SEED = 42  # Fixed seed for reproducibility
CHUNK = 250  # How many resamples to draw at once


@dataclass(frozen=True)
class Units:
    """A set of units to draw from, each with a total and a weight."""

    totals: np.ndarray
    weights: np.ndarray

    @classmethod
    def of_rows(cls, values: np.ndarray) -> Self:
        """Take each row as its own unit, which is the ordinary bootstrap.

        Args:
            values: One row per unit, `[units, ...]`, e.g. the prefixes of one length scored on
                every metric.
        Returns:
            Those rows at weight 1 each, so the ratio estimator reduces to their plain mean.
        """
        return cls(totals=values, weights=np.ones(len(values), dtype=np.float64))

    @classmethod
    def of_clusters(cls, totals: np.ndarray, weights: np.ndarray) -> Self:
        """Take each cluster as one unit, which is the cluster bootstrap.
        Args:
            totals: Each cluster's summed scores, `[clusters, ...]`.
            weights: How many rows each cluster summed, `[clusters]`.
        Returns:
            Those clusters, drawn whole.
        """
        return cls(totals=totals, weights=weights.astype(np.float64))

    def __len__(self) -> int:
        return len(self.weights)

    @property
    def mean(self) -> np.ndarray:
        """The observed mean these units were drawn around, `[...]`: the mean a report holds."""
        return self.totals.sum(axis=0) / self.weights.sum()


def _resample_counts(units: int, resamples: int, *, generator: np.random.Generator) -> Iterator:
    """Draw a bootstrap's resamples, a chunk of them at a time.

    Args:
        units: How many units there are to draw from.
        resamples: How many resamples to draw in total.
        generator: The source of the draws, seeded by the caller off `SEED`.
    Yields:
        `[chunk, units]` of counts as float64, ready to be multiplied against the units' values.
        The chunks are `CHUNK` long except the last, and sum to `resamples`.
    """
    drawn = 0
    while drawn < resamples:
        size = min(CHUNK, resamples - drawn)
        picks = generator.integers(low=0, high=units, size=(size, units))
        picks += (np.arange(size) * units)[:, None]
        yield (
            np.bincount(picks.ravel(), minlength=size * units)
            .reshape(size, units)
            .astype(np.float64)
        )
        drawn += size


def resample_means(
    units: Units, resamples: int, *, generator: np.random.Generator
) -> Iterator[np.ndarray]:
    """Draw a bootstrap's resample means, a chunk of them at a time.

    The one place a resample is turned into a number in this package. What a caller does with the
    stream is what separates a confidence interval from a p-value: `intervals` reads two order
    statistics off it, `significance` counts which side of a reference each mean fell.

    Args:
        units: What is being drawn from.
        resamples: How many resamples to draw in total.
        generator: The source of the draws, seeded by the caller off `SEED`.
    Yields:
        `[chunk, *totals.shape[1:]]` of resample means. One draw serves every trailing axis at
        once: the counts are `[chunk, units]` and the totals `[units, ...]`, so a resample's means
        are one matmul rather than a pass per metric.
    """
    flat = units.totals.reshape(len(units), -1)
    for counts in _resample_counts(len(units), resamples, generator=generator):
        means = (counts @ flat) / (counts @ units.weights)[:, None]
        yield means.reshape(len(counts), *units.totals.shape[1:])
